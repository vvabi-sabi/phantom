// Inverted-hull outline: the vertex is pushed along its normal in screen
// space by a thickness given in pixels; flat color, unlit.

#import bevy_pbr::{
    mesh_view_bindings::view,
    mesh_bindings::mesh,
    mesh_functions,
    skinning,
    forward_io::Vertex,
}

struct OutlineMaterial {
    color: vec4<f32>,
    thickness_px: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var<uniform> material: OutlineMaterial;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    var out: VertexOutput;

#ifdef SKINNED
    let world_from_local = skinning::skin_model(
        vertex_in.joint_indices,
        vertex_in.joint_weights,
        vertex_in.instance_index,
    );
    let world_normal = skinning::skin_normals(world_from_local, vertex_in.normal);
#else
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    let world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex_in.normal,
        vertex_in.instance_index,
    );
#endif

    let world_pos = (world_from_local * vec4(vertex_in.position, 1.0)).xyz;
    var clip_pos = view.clip_from_world * vec4(world_pos, 1.0);

    // Shift along the normal in screen space by thickness_px pixels.
    let view_normal = normalize((view.view_from_world * vec4(world_normal, 0.0)).xyz);
    var clip_normal = (view.clip_from_view * vec4(view_normal, 0.0)).xy;
    clip_normal = normalize(clip_normal + vec2(1e-6, 0.0));
    let ndc_offset = 2.0 * material.thickness_px / view.viewport.zw;
    let shifted = clip_pos.xy + clip_normal * ndc_offset * clip_pos.w;
    clip_pos = vec4<f32>(shifted, clip_pos.z, clip_pos.w);

    out.clip_position = clip_pos;
    return out;
}

@fragment
fn fragment() -> @location(0) vec4<f32> {
    return material.color;
}
