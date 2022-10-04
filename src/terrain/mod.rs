use std::collections::HashMap;

use bevy::{
    math::{UVec2, Vec2, Vec3},
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
};
use image::{ImageBuffer, Luma};

pub mod terrain_pipeline;

pub type Triangle = (UVec2, UVec2, UVec2);
type HeightMap = ImageBuffer<Luma<u16>, Vec<u16>>;
pub struct TerrainMeshData {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<u32>,
}

#[derive(Eq, PartialEq, Debug)]
pub enum TrType {
    TopRight,
    BottomLeft,
    Left,
    Right,
}

pub struct RTIN {
    pub error_threshold: f32,
    pub pixel_side_length: f32,
    pub max_image_height: f32,
}

impl RTIN {
    pub fn get_index_level_start(level: u32) -> u32 {
        ((2 << level) - 1) & (!1u32)
    }

    pub fn index_to_bin_id(index: u32) -> u32 {
        let mut level = 0;
        let mut index_level_start = 0;

        for i in 0..32 {
            let new_index_level_start = RTIN::get_index_level_start(i);
            if index >= new_index_level_start {
                level = i;
                index_level_start = new_index_level_start;
            } else {
                break;
            }
        }

        (1 << (level + 1)) + (index - index_level_start)
    }

    fn most_sig_bit(num: u32) -> u32 {
        32 - num.leading_zeros()
    }

    pub fn id_to_index_in_level(bin_id: u32) -> u32 {
        bin_id - (1 << (Self::most_sig_bit(bin_id) - 1))
    }

    pub fn id_to_level(bin_id: u32) -> u32 {
        Self::most_sig_bit(bin_id) - 2
    }

    pub fn id_to_step(bin_id: u32) -> Vec<TrType> {
        let mut steps = Vec::new();
        let triangle_level = Self::id_to_level(bin_id);

        if bin_id & 1 > 0 {
            steps.push(TrType::TopRight);
        } else {
            steps.push(TrType::BottomLeft);
        }

        for i in 1..(triangle_level + 1) {
            if bin_id & (1 << i) > 0 {
                steps.push(TrType::Left);
            } else {
                steps.push(TrType::Right);
            }
        }

        steps
    }

    pub fn get_tr_coords(bin_id: u32, grid_size: u32) -> Triangle {
        let mut a = UVec2::new(0, 0);
        let mut b = UVec2::new(0, 0);
        let mut c = UVec2::new(0, 0);

        for step in Self::id_to_step(bin_id) {
            match step {
                TrType::TopRight => {
                    // north east right-angle corner
                    a[0] = 0;
                    a[1] = 0;
                    b[0] = grid_size - 1;
                    b[1] = grid_size - 1;
                    c[0] = grid_size - 1;
                    c[1] = 0;
                }
                TrType::BottomLeft => {
                    // north east right-angle corner
                    a[0] = grid_size - 1;
                    a[1] = grid_size - 1;
                    b[0] = 0;
                    b[1] = 0;
                    c[0] = 0;
                    c[1] = grid_size - 1;
                }
                TrType::Left => {
                    let (new_a, new_b, new_c) = (c, a, (a + b) / 2);
                    a = new_a;
                    b = new_b;
                    c = new_c;
                }
                TrType::Right => {
                    let (new_a, new_b, new_c) = (b, c, (a + b) / 2);
                    a = new_a;
                    b = new_b;
                    c = new_c;
                }
            }
        }

        (a, b, c)
    }

    pub fn pixel_for_tr_mid_point(bin_id: u32, grid_size: u32) -> UVec2 {
        let triangle_coords = Self::get_tr_coords(bin_id, grid_size);
        let mid_point = (triangle_coords.0 + triangle_coords.1) / 2;

        UVec2::new(mid_point[0], mid_point[1])
    }

    pub fn corner_mean(heightmap: &HeightMap, corner_u32: UVec2) -> f32 {
        let mut new_corner = corner_u32;

        if new_corner[0] >= heightmap.width() {
            new_corner[0] = heightmap.width() - 1;
        }

        if new_corner[1] >= heightmap.height() {
            new_corner[1] = heightmap.height() - 1;
        }

        heightmap.get_pixel(new_corner[0], new_corner[1]).0[0] as f32 / std::u16::MAX as f32
    }

    pub fn errors_vec_index(bin_id: u32, grid_size: u32) -> usize {
        let triangle_midpoint = Self::pixel_for_tr_mid_point(bin_id, grid_size);
        let midpoint_error_vec_index = triangle_midpoint[1] * grid_size + triangle_midpoint[0];

        midpoint_error_vec_index as usize
    }

    pub fn id_to_index(bin_id: u32) -> u32 {
        let level = Self::id_to_level(bin_id);
        let index_level_start = Self::get_index_level_start(level);
        let index_in_level = Self::id_to_index_in_level(bin_id);

        index_level_start + index_in_level
    }

    pub fn get_children_ids(bin_id: u32) -> (u32, u32) {
        let level = Self::id_to_level(bin_id);
        let right_bin_id = bin_id + (1 << (level + 2)) - (1 << (level + 1));
        let left_bin_id = bin_id + (1 << (level + 2));
        (right_bin_id, left_bin_id)
    }

    pub fn get_children_indices(bin_id: u32) -> (u32, u32) {
        let (right_index, left_index) = Self::get_children_ids(bin_id);
        (
            Self::id_to_index(right_index),
            Self::id_to_index(left_index),
        )
    }

    pub fn build_errors_vec(&self, heightmap: &HeightMap) -> Vec<f32> {
        assert_eq!(heightmap.width(), heightmap.height());
        assert!(heightmap.width().is_power_of_two());

        let side = heightmap.width();
        let grid_size = side + 1;
        let number_of_triangles = side * side * 2 - 2;
        let number_of_levels = side.log2() * 2;
        let last_level = number_of_levels - 1;

        let last_level_index_start = Self::get_index_level_start(last_level);

        let mut errors_vec = Vec::new();
        errors_vec.resize((grid_size * grid_size) as usize, 0.0f32);

        for triangle_index in (0..number_of_triangles).rev() {
            let triangle_bin_id = Self::index_to_bin_id(triangle_index);

            let midpoint = Self::pixel_for_tr_mid_point(triangle_bin_id, grid_size);

            let triangle_coords = Self::get_tr_coords(triangle_bin_id, grid_size);
            let h0 = Self::corner_mean(heightmap, triangle_coords.0);
            let h1 = Self::corner_mean(heightmap, triangle_coords.1);
            let midpoint_interpolated = (h1 + h0) / 2.0;
            let midpoint_height = Self::corner_mean(heightmap, midpoint);

            let this_triangle_error = (midpoint_interpolated - midpoint_height).abs();

            let this_triangle_mid_point_error_vec_index =
                Self::errors_vec_index(triangle_bin_id, grid_size);

            // println!("Processing triangle {:b} of coords {:?} with error index {}",
            //      triangle_bin_id, triangle_coords, this_triangle_mid_point_error_vec_index);

            if triangle_index >= last_level_index_start {
                errors_vec[this_triangle_mid_point_error_vec_index] = this_triangle_error;
            } else {
                let (right_child_bin_id, left_child_bin_id) =
                    Self::get_children_ids(triangle_bin_id);

                let right_errors_vec_index = Self::errors_vec_index(right_child_bin_id, grid_size);
                let left_errors_vec_index = Self::errors_vec_index(left_child_bin_id, grid_size);

                let prev_error = errors_vec[this_triangle_mid_point_error_vec_index];
                let right_error = errors_vec[right_errors_vec_index];
                let left_error = errors_vec[left_errors_vec_index];

                errors_vec[this_triangle_mid_point_error_vec_index] = prev_error
                    .max(left_error)
                    .max(right_error)
                    .max(this_triangle_error);
            }
        }

        errors_vec
    }

    pub fn select_triangles_scoped(
        &self,
        heightmap: &HeightMap,
        errors_vec: &Vec<f32>,
        triangles: &mut Vec<u32>,
        triangle_index: u32,
    ) {
        let grid_size = heightmap.width() + 1;

        let triangle_bin_id = Self::index_to_bin_id(triangle_index);

        let (right_child_index, left_child_index) = Self::get_children_indices(triangle_bin_id);

        let side = heightmap.width();
        let number_of_last_level_triangles = side * side * 2;
        let number_of_triangles = side * side * 2 - 2 + number_of_last_level_triangles;

        let has_children = right_child_index < number_of_triangles;

        let leaf_triangle = !has_children;

        let this_triangle_errors_vec_index = Self::errors_vec_index(triangle_bin_id, grid_size);
        let this_triangle_error = errors_vec[this_triangle_errors_vec_index];
        let error_within_threshold = this_triangle_error <= self.error_threshold;

        if error_within_threshold || leaf_triangle {
            triangles.push(triangle_bin_id);
        } else {
            self.select_triangles_scoped(heightmap, errors_vec, triangles, left_child_index);
            self.select_triangles_scoped(heightmap, errors_vec, triangles, right_child_index);
        }
    }

    pub fn select_triangles(&self, heightmap: &HeightMap, errors_vec: &Vec<f32>) -> Vec<u32> {
        let mut triangles = Vec::<u32>::new();

        self.select_triangles_scoped(heightmap, &errors_vec, &mut triangles, 0);
        self.select_triangles_scoped(heightmap, &errors_vec, &mut triangles, 1);

        triangles
    }

    pub fn build_terrain(&self, heightmap: &HeightMap) -> TerrainMeshData {
        let errors_vec = self.build_errors_vec(heightmap);

        let mut vertices = Vec::<Vec3>::new();
        let mut indices = Vec::<u32>::new();
        let mut vertices_array_position = HashMap::<u32, usize>::new();

        let triangle_bin_ids = self.select_triangles(heightmap, &errors_vec);

        for triangle_bin_id in triangle_bin_ids {
            let grid_size = heightmap.width() + 1;
            let triangle_coords = Self::get_tr_coords(triangle_bin_id, grid_size);
            let new_vertices = &[triangle_coords.0, triangle_coords.1, triangle_coords.2];

            for new_vertex in new_vertices {
                let vertex_id = new_vertex[1] * grid_size + new_vertex[0];

                let vertex_index = if vertices_array_position.contains_key(&vertex_id) {
                    *vertices_array_position.get(&vertex_id).unwrap()
                } else {
                    let new_vertex_index = vertices.len();
                    vertices_array_position.insert(vertex_id, new_vertex_index);

                    let vertex_height = Self::corner_mean(heightmap, *new_vertex);

                    let new_vertex_3d =
                        Vec3::new(new_vertex[0] as f32, vertex_height, new_vertex[1] as f32);
                    vertices.push(new_vertex_3d);
                    new_vertex_index
                };
                indices.push(vertex_index as u32);
            }
        }

        TerrainMeshData { vertices, indices }
    }

    pub fn build_mesh(&self, terrain_mesh_data: &TerrainMeshData, heightmap: &HeightMap) -> Mesh {
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();

        let mut indices: Vec<u32> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();
        let indices_len = terrain_mesh_data.indices.len();

        vertices.reserve(terrain_mesh_data.vertices.len());
        normals.reserve(terrain_mesh_data.vertices.len());
        uvs.reserve(terrain_mesh_data.vertices.len());

        //normals.resize(terrain_mesh_data.vertices.len(), [0.0,1.0,0.0]);
        //colors.reserve(vertices.len());
        indices.reserve(indices_len);

        //let grad = Gradient::new(vec![
        //    Hsv::from(LinSrgb::new(1.0, 0.1, 0.1)),
        //    Hsv::from(LinSrgb::new(0.1, 1.0, 1.0)),
        //]);

        let size = (heightmap.width() + 1) as f32;

        for vertex in &terrain_mesh_data.vertices {
            vertices.push([
                vertex.x * self.pixel_side_length,
                vertex.y * self.max_image_height,
                vertex.z * self.pixel_side_length,
            ]);

            //let color = grad.get(vertex.y);
            //let raw_float: Srgb<f32> = Srgb::<f32>::from_linear(color.into());
            uvs.push([vertex.x as f32 / size, vertex.z as f32 / size]);
            normals.push([0.0, 1.0, 0.0]);
            //let c = vertex.y;
            let w = heightmap.width() - 1;

            let color = match vertex.y {
                y if y == 0.0 => [0.0, 0.0, 1.0, 1.0],
                y if y < 0.1 => [1.0, 0.70, 0.5, 1.0],
                y => {
                    let h = heightmap
                        .get_pixel((vertex.x as u32).min(w), (vertex.z as u32).min(w))
                        .0[0] as f32
                        / std::u16::MAX as f32;
                    let opacity = if y < 0.3 { 0.0 } else { 1.0 };
                    [opacity, 0.0, 0.0, opacity]
                }
                
            };
            colors.push(color);
        }

        let triangle_number = terrain_mesh_data.indices.len() / 3;

        let mut max = 0;
        for i in 0..triangle_number {
            for j in 0..3 {
                indices.push(terrain_mesh_data.indices[i * 3 + j]);
                max = max.max(terrain_mesh_data.indices[i * 3 + j]);
            }
        }

        println!(
            "{} {} {} {}",
            vertices.len(),
            normals.len(),
            uvs.len(),
            indices.len()
        );

        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(terrain_pipeline::ATTRIBUTE_BLEND_COLOR, colors);
        mesh.set_indices(Some(Indices::U32(indices)));

        mesh
    }

    pub fn load_terrain(&self, filename: &str) -> Mesh {
        let terrain_image = image::open(filename).unwrap();
        let terrain_heightmap = terrain_image.as_luma16().unwrap();
        let terrain_mesh_data = self.build_terrain(terrain_heightmap);

        let shaded_mesh = self.build_mesh(&terrain_mesh_data, &terrain_heightmap);

        shaded_mesh
    }
}

impl Default for RTIN {
    fn default() -> Self {
        RTIN {
            error_threshold: 0.2,
            pixel_side_length: 1.0,
            max_image_height: 1.0,
        }
    }
}
