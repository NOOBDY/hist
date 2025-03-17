#version 460

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform UB {
    vec4 color;
} ub;

void main() {
    f_color = vec4(uv, 0.0, 1.0);
    // f_color = ub.color;
}
