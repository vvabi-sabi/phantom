//! Outline: a separate unlit pass using an inverted hull.
//! Ignores lighting; thickness is given in output pixels.

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    mesh::MeshVertexBufferLayoutRef,
    pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin},
    prelude::*,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
};

pub const OUTLINE_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("f1a2b3c4-d5e6-47a8-b9c0-d1e2f3a4b5c6");

pub struct OutlinePlugin;

impl Plugin for OutlinePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            OUTLINE_SHADER_HANDLE,
            "outline.wgsl",
            Shader::from_wgsl
        );
        app.add_plugins(MaterialPlugin::<OutlineMaterial>::default());
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct OutlineMaterial {
    #[uniform(0)]
    pub color: LinearRgba,
    #[uniform(0)]
    pub thickness_px: f32,
}

impl Material for OutlineMaterial {
    fn vertex_shader() -> ShaderRef {
        OUTLINE_SHADER_HANDLE.into()
    }

    fn fragment_shader() -> ShaderRef {
        OUTLINE_SHADER_HANDLE.into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<OutlineMaterial>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The outline applies only to characters' skinned meshes.
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
            Mesh::ATTRIBUTE_JOINT_INDEX.at_shader_location(6),
            Mesh::ATTRIBUTE_JOINT_WEIGHT.at_shader_location(7),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        // Inverted hull: draw only back faces.
        descriptor.primitive.cull_mode = Some(bevy::render::render_resource::Face::Front);
        Ok(())
    }
}
