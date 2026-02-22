
from manimlib import *
import numpy as np
import sympy as sp

class ShapeOperator(ThreeDScene):
    def surface_map(self, u, v):
        new_result = np.array([
            u*np.cos(v),
            u*np.sin(v),
            np.sin((u*np.cos(v))**3) + np.cos((u*np.sin(v))**3)
        ])
        return new_result

    def f_derivatives(self, u_val, v_val):
        u, v = sp.symbols('u v')
        f = sp.Matrix([
            u*sp.cos(v),
            u*sp.sin(v),
            sp.sin((u*sp.cos(v))**3) + sp.cos((u*sp.sin(v))**3)
        ])

        f_u = f.diff(u)
        f_v = f.diff(v)

        f_u_eval = f_u.subs({u: u_val, v: v_val}).evalf()
        f_v_eval = f_v.subs({u: u_val, v: v_val}).evalf()

        return (f_u_eval, f_v_eval)

    def construct(self):
        self.camera.frame.set_euler_angles(phi=70*DEGREES, theta=160*DEGREES)
        self.camera.frame.set_height(6)
        axes = ThreeDAxes(x_range=[-2,2,1], y_range=[-2,2,1], z_range=[-2,2,1])
        self.play(ShowCreation(axes))

        # surface parametrization axes label
        surface_param_label = Tex(r"Surface \text{ } Parametrization").move_to(np.array([0,3,0]))
        surface_param_label.scale(0.5)
        surface_param_label.set_color(WHITE)
        surface_param_label.rotate(90*DEGREES, axis=RIGHT)
        surface_param_label.rotate(180*DEGREES)
        self.play(Write(surface_param_label))

        # U circle
        circle = Circle(radius=1.0, arc_center=np.array([0,0,0]))
        circle.set_fill(BLUE, opacity=0)
        circle.set_stroke(BLUE_E, width=4)
        circle_label = Tex("U", color=BLUE).move_to(np.array([1.1, 1.1, 0]))
        circle_label.set_color(BLUE)
        circle_label.rotate(90*DEGREES, axis=RIGHT)
        circle_label.rotate(180*DEGREES)


        self.play(ShowCreation(circle), Write(circle_label)) # draw vectors u, v on U (circle) 
        point_pos = np.array([0.5, 0.5, 0])

        U_point = Dot(point=point_pos)
        U_point.set_color(GREEN)
        self.add(U_point)

        self.wait(1)
        
        # surface
        surface = ParametricSurface(
                lambda u, v: np.array([
                    u*np.cos(v),
                    u*np.sin(v),
                    np.sin((u*np.cos(v))**3) + np.cos((u*np.sin(v))**3)
                ]),
                u_range = [0,1],
                v_range = [0, 2*np.pi],
                resolution=(20,40))
        surface.set_color(ORANGE)
        surface.set_opacity(0.6)

        self.play(ShowCreation(surface))

        # zooming in
        # self.play(
        #         self.camera.frame.animate.set_height(5),
        #         run_time=2
        # )

        # get the point on the surface that our U_point is sent to
        surface_point = self.surface_map(point_pos[0], point_pos[1])
        surface_pt = Dot(point=surface_point)
        surface_pt.set_color(GREEN)

        # map an arrow from U_point to our future surface_point
        f_arrow = Arrow(point_pos, surface_point)
        f_arrow.set_color(GREEN)
        self.play(GrowArrow(f_arrow))
        self.add(surface_pt)

        # rotate our camera
        self.play(self.frame.animate.set_euler_angles(
            phi=70*DEGREES,
            theta=190*DEGREES),
                  run_time=2
        )

        # write our parameterization
        f = r"f: "
        U = r"U "
        right_arrow = r"\rightarrow "
        R3 = r"\mathbb{R}^{3}"
        param_label = Tex(f, U, right_arrow, R3).move_to(np.array([0,0,1]))
        param_label.rotate(90*DEGREES, axis=RIGHT)
        param_label.rotate(180*DEGREES)
        param_label.shift(5*RIGHT + 6*DOWN)

        param_label[0].set_color(GREEN)
        param_label[2].set_color(BLUE)
        param_label[4].set_color(ORANGE)
        param_label[5].set_color(ORANGE)
        param_label.scale(1.25)
        self.play(Write(param_label))

        self.wait(1)

        # I just want to see what df/du and df/dv vectors at our surface_point produce
        df_du, df_dv = self.f_derivatives(point_pos[0], point_pos[1])
        df_du = np.array(df_du.tolist(), dtype=float).flatten()
        df_dv = np.array(df_dv.tolist(), dtype=float).flatten()

        # Computing the normal vector to the surface_point
        numerator = np.cross(df_du, df_dv)
        denom = np.linalg.norm(numerator)
        nu = numerator/denom # normal vector
        nu = nu/np.linalg.norm(nu)
        # nu_arrow = Arrow(surface_point, surface_point + nu)
        # nu_arrow.set_color(PURPLE)
        # self.play(GrowArrow(nu_arrow))

        nu_line = Line(surface_point, surface_point + nu)
        nu_line.set_color(BLUE_B)
        nu_line.set_stroke(width=5)
        self.play(ShowCreation(nu_line))

        nu_label = Tex(r"\hat{n}").move_to(surface_point + 1.4*nu)
        nu_label.set_color(BLUE_B)
        nu_label.rotate(90*DEGREES, axis=RIGHT)
        nu_label.rotate(180*DEGREES)
        nu_label.scale(0.75)
        self.play(Write(nu_label))

        # zoom out

        # create the Gauss map axes
        gauss_axes = ThreeDAxes(x_range=[-2,2,1], y_range=[-2,2,1], z_range=[-2,2,1])

        self.play(self.camera.frame.animate.scale(1.25))
        self.play(self.camera.frame.animate.shift(LEFT*3))

        self.wait(0.25)

        self.play(ShowCreation(gauss_axes))
        self.play(
                gauss_axes.animate.shift(LEFT*5)
        )

        gauss_label = Tex(r"Gauss \text{ } Map").move_to(np.array([-6, 3, 0]))
        gauss_label.set_color(WHITE)
        gauss_label.rotate(90*DEGREES, axis=RIGHT)
        gauss_label.rotate(180*DEGREES)
        gauss_label.scale(0.5)
        self.play(Write(gauss_label))

        # draw S^2 - the unit sphere of the gauss map
        gauss_axes_origin = gauss_axes.get_origin()
        wireframe = VGroup()
        for angle in np.linspace(0, PI, 14):
            ring = Circle(radius=1).rotate(angle, axis=UP)
            ring.move_to(gauss_axes_origin)
            wireframe.add(ring)
        for angle in np.linspace(0, PI, 14):
            ring = Circle(radius=1).rotate(angle, axis=RIGHT)
            ring.move_to(gauss_axes_origin)
            wireframe.add(ring)
        wireframe.set_stroke(GREY, width=1, opacity=0.5)
        self.play(ShowCreation(wireframe))

        # parallel translation: sending nu to the origin of the gauss map
        nu_group = VGroup(nu_line, nu_label).copy()
        shift_vec = gauss_axes.get_origin() - surface_point
        self.play(nu_group.animate.shift(shift_vec))

        # build the tangent planes
        # tangent plane to the surface point
        tangent_plane = ParametricSurface(
                lambda s,t: surface_point + s*df_du + t*df_dv,
                u_range = (-1,1),
                v_range = (-1,1),
                resolution = (10,10)
        )
        tangent_plane.shift(0.01*nu)

        tangent_plane.set_color(PINK)
        tangent_plane.set_opacity(0.8)
        
        tangent_plane_label = Tex(r"T_{p}U").move_to(tangent_plane.get_left())
        tangent_plane_label.set_color(PINK)
        tangent_plane_label.rotate(90*DEGREES, axis=RIGHT)
        tangent_plane_label.rotate(180*DEGREES)
        tangent_plane_label.scale(0.75)
        tangent_plane_label.shift(1*LEFT)

        self.play(ShowCreation(tangent_plane))

        self.wait(1)

        # tangent_plane to the gauss map translation
        gtp = tangent_plane.copy()
        shift_tp = (gauss_axes_origin + nu) - tangent_plane.get_center()
        self.play(gtp.animate.shift(shift_tp))


        gtp_label = Tex(r"T_{\nu(p)} \mathbf{R}^{3} = T_{p}U").move_to(gtp.get_right())
        gtp_label.set_color(PINK)
        gtp_label.rotate(90*DEGREES, axis=RIGHT)
        gtp_label.rotate(180*DEGREES)
        gtp_label.scale(0.75)
        gtp_label.shift(3*LEFT)

        self.wait(1)


        self.play(Write(tangent_plane_label), Write(gtp_label))

        self.wait(1)

        # introduce the shape operator
        # draw the shape operator label

        # starting moving the tangent planes around the surfaces with normal vector connected

        # these labels feel too cluttered
        # normal_label = r"\mathbf{N}"
        # vf_label = r" - unit normal vector field"
        # param_label = Tex(normal_label, vf_label).move_to(ORIGIN + 1.5*LEFT + 12*DOWN)
        # normal_field_label.set_color(WHITE)
        # normal_field_label.rotate(90*DEGREES, axis=RIGHT)
        # normal_field_label.rotate(180*DEGREES)
        # normal_field_label.scale(0.75)


        shape_op_label = Tex(r"S_{p}(v) = - D_{v} \mathbf{N}").move_to(ORIGIN + 1*LEFT + 12*DOWN)
        shape_op_label.set_color(WHITE)
        shape_op_label.rotate(90*DEGREES, axis=RIGHT)
        shape_op_label.rotate(180*DEGREES)
        shape_op_label.scale(1.5)

        self.play(Write(shape_op_label))

        self.wait(1)

        # Animate nu and the tangent plane to move along the surface
        # shows shape operator
        u_fixed = point_pos[0]
        v_start = point_pos[1]
        v_end = v_start + 2*PI
        gm_origin = gauss_axes.get_origin()

        def numerical_deriv(u_val, v_val, eps=1e-5):
            # deriv = move a little bit each direction, see the difference
            du = (self.surface_map(u_val + eps, v_val) - self.surface_map(u_val - eps, v_val))/(2*eps)
            dv = (self.surface_map(u_val, v_val + eps) - self.surface_map(u_val, v_val - eps))/(2*eps)
            return du, dv

        def get_data(v_val):
            surface_point = self.surface_map(u_fixed, v_val)
            du, dv = numerical_deriv(u_fixed, v_val)
            n = np.cross(du, dv)
            n = n/np.linalg.norm(n)
            return surface_point, du, dv, n


        def update_all(mob, alpha):
            v_val = v_start + alpha*(v_end - v_start)
            sp, du, dv, n = get_data(v_val)

            nu_line.become(
                    Line(sp, sp+n).set_color(BLUE_B).set_stroke(width=5)
            )

            new_tp = ParametricSurface(
                    lambda s, t, _sp = sp, _du=du, _dv=dv: _sp + s*_du + t*_dv,
                    u_range=(-1,1), v_range=(-1,1), resolution=(10,10)
            )
            new_tp.set_color(PINK).set_opacity(0.8)
            tangent_plane.become(new_tp)

            # gauss map
            nu_group[0].become(
                    Line(gm_origin, gm_origin + n).set_color(BLUE_B).set_stroke(width=5)
            )

            new_gtp = ParametricSurface(
                    lambda s, t, _base=gm_origin+n, _du=du, _dv=dv: _base + s*_du + t*_dv,
                    u_range=(-1,1), v_range=(-1,1), resolution=(10,10)
            )
            new_gtp.set_color(PINK).set_opacity(0.8)
            gtp.become(new_gtp)

            nu_label.move_to(sp + 1.4*n)
            nu_group[1].move_to(gm_origin + 1.4*n)

        dummy = VMobject()
        self.play(UpdateFromAlphaFunc(dummy, update_all), run_time=10)
        

        self.wait()
        self.embed() # make manimgl interactive
