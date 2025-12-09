from typing import Tuple, Union

import moderngl
import numpy as np


class Renderer:
    """! Simple OpenGL offscreen renderer using ModernGL."""

    def __init__(
        self,
        size: Tuple[int, int],
        samples: int = 0,
        point_size: float = 1.0,
        line_width: float = 1.0,
    ) -> None:
        """! Class initializer.

        @param size Image size (width, height).
        @param samples Number of samples for multisampling.
        @param point_size Point size.
        @param line_width Line width.
        """
        self.ctx = moderngl.create_standalone_context()
        self.ctx.gc_mode = "auto"

        self.ctx.point_size = point_size
        self.ctx.line_width = line_width

        self.program = self.ctx.program(
            vertex_shader="""
            #version 330 core

            uniform mat4 world_to_clip;

            layout (location = 0) in vec3 in_vertex;
            layout (location = 1) in vec3 in_color;

            out vec3 color;

            void main() {
                gl_Position = world_to_clip * vec4(in_vertex, 1.0);
                color = in_color;
            }
            """,
            fragment_shader="""
            #version 330 core

            uniform bool normalize_color;

            in vec3 color;

            out vec3 frag_color;

            void main()
            {
                if (normalize_color)
                    frag_color = normalize(color);
                else
                    frag_color = color;
            }
            """,
        )

        self.depth_texture = self.ctx.depth_texture(size, alignment=1, samples=samples)
        self.fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(size, components=4, samples=samples, dtype="f4"),
            depth_attachment=self.depth_texture,
        )
        self.output = self.ctx.framebuffer([self.ctx.renderbuffer(size, components=4, dtype="f4")])

        self.vaos = list()

    def add(self, verts: np.ndarray, colors: np.ndarray, primitive: int) -> None:
        """! Add an object to the scene.

        @param verts Vertices (N, 3).
        @param colors Per-vertex colors (N, 3).
        @param primitive Primitive type (e.g. moderngl.POINTS).
        """
        vbo_verts = self.ctx.buffer(verts.astype("f4").tobytes())
        vbo_colors = self.ctx.buffer(colors.astype("f4").tobytes())
        vao = self.ctx.vertex_array(self.program, [(vbo_verts, "3f", "in_vertex"), (vbo_colors, "3f", "in_color")])
        self.vaos.append((vao, primitive))

    def clear(self) -> None:
        """! Remove all objects from the scene."""
        for vao, _ in self.vaos:
            vao.release()
        self.vaos.clear()

    def render(
        self, world_to_view: np.ndarray, view_to_clip: np.ndarray, return_depth: bool = False, as_float: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """! Render the scene.

        @param world_to_view World-to-view transformation (4, 4).
        @param view_to_clip View-to-clip space transformation (4, 4).
        @param return_depth Whether to return a depth image in addition to the RGB image.
        @param as_float Whether to return the RGB image in floating point format.
        @return Returns the rendered RGB image, and the rendered depth if return_depth is set.
        """
        self.fbo.use()
        self.ctx.clear()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        world_to_clip = view_to_clip @ world_to_view
        self.program["world_to_clip"].write(np.ascontiguousarray(world_to_clip.T).astype(np.float32))

        for vao, primitive in self.vaos:
            vao.render(primitive)

        self.ctx.copy_framebuffer(self.output, self.fbo)
        data = self.output.read(dtype="f4" if as_float else "f1")
        img = np.frombuffer(data, dtype="f4" if as_float else "uint8").reshape((*self.fbo.size[1::-1], 3))
        img = np.flipud(img)

        if return_depth:
            depth_data = self.depth_texture.read(alignment=1)
            depth_img = np.frombuffer(depth_data, dtype="f4").reshape(self.fbo.size[1::-1])
            # See https://community.khronos.org/t/ndc-depth-to-pre-perspective-divide-depth/60758
            depth_img = -view_to_clip[2, 3] / (depth_img * -2.0 + 1.0 - view_to_clip[2, 2])
            depth_img = np.flipud(depth_img)
            return img, depth_img
        else:
            return img
