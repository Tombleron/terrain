#![feature(int_log)]
use bevy::{prelude::*, window::PresentMode, winit::WinitSettings};
use bevy_egui::{egui, EguiContext, EguiPlugin};
use smooth_bevy_cameras::{
    controllers::unreal::{UnrealCameraBundle, UnrealCameraController, UnrealCameraPlugin},
    LookTransformPlugin,
};

mod terrain;
use terrain::terrain_pipeline::TerrainMaterial;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(LookTransformPlugin)
        .add_plugin(UnrealCameraPlugin::default())
        .add_plugin(MaterialPlugin::<TerrainMaterial>::default())
        // Optimal power saving and present mode settings for desktop apps.
        .insert_resource(WinitSettings::desktop_app())
        .insert_resource(WindowDescriptor {
            present_mode: PresentMode::Mailbox,
            ..Default::default()
        })
        .add_startup_system(setup_system)
        //.add_system(ui_example_system)
        .run();
}

#[derive(Default, Component)]
struct Marker;

fn setup_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cus_materials: ResMut<Assets<TerrainMaterial>>,

    asset_server: Res<AssetServer>,
) {
    //commands.spawn_bundle(PbrBundle {
    //    mesh: meshes.add(Mesh::from(shape::Plane { size: 5.0 })),
    //    material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
    //    ..Default::default()
    //});
    //commands
    //    .spawn_bundle(PbrBundle {
    //        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
    //        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
    //        transform: Transform::from_xyz(0.0, 0.5, 0.0),
    //        ..Default::default()
    //    })
    //    .insert(Marker);
    //commands.spawn_bundle(PointLightBundle {
    //    point_light: PointLight {
    //        intensity: 1500.0,
    //        shadows_enabled: true,
    //        ..Default::default()
    //    },
    //    transform: Transform::from_xyz(4.0, 8.0, 4.0),
    //    ..Default::default()
    //});

    let rtin = terrain::RTIN {
        error_threshold: 0.0000001,
        pixel_side_length: 1.0,
        max_image_height: 50.0,
    };
    let terr_mesh = rtin.load_terrain("assets/aus.png");
    let terr_mesh_handle = meshes.add(terr_mesh);

    //let texture_handle = asset_server.load("aust.png");

    commands
        .spawn_bundle(MaterialMeshBundle {
            mesh: terr_mesh_handle,
            material: cus_materials.add(TerrainMaterial {
                color: Color::rgba(1.0, 1.0, 1.0, 1.0),
            }),
            //material: materials.add(StandardMaterial{
            //    base_color: Color::rgba(1.0, 1.0, 1.0, 1.0),
            //    base_color_texture: Some(texture_handle.clone()),
            //    alpha_mode: AlphaMode::Blend,
            //    ..Default::default()
            //}),
            transform: Transform::from_xyz(0.0, 0.5, 0.0),
            ..Default::default()
        })
        .insert(Marker);

    println!("seks7");
    commands.spawn_bundle(UnrealCameraBundle::new(
        UnrealCameraController::default(),
        PerspectiveCameraBundle::default(),
        Vec3::new(20.0, 300.0, 20.0),
        Vec3::new(20., 0., 20.),
    ));
}
