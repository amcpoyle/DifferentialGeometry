from manimlib import *
import numpy as np
import sympy as sp

class CurvatureComparison(ThreeDScene):
    def eval_curve(self, t):
        return np.array([t, 0.1*t**2, 0])

    def eval_surface(self, u,v):
        pass

    def curve_normal(self, t_val):
        t = sp.symbols('t')
        f = sp.Matrix([
            t,
            0.1*t**2,
            0
        ])

        df_dt = f.diff(t)
        df_dt2 = df_dt.diff(t)
        s_prime = sp.sqrt(df_dt.dot(df_dt))

        normal_vector = ((df_dt2*(s_prime**2)) - df_dt*(df_dt.dot(df_dt2)))/(s_prime*sp.sqrt((df_dt2.dot(df_dt2))*(s_prime**2) - (df_dt.dot(df_dt2))**2))
        normal_vector = normal_vector/normal_vector.norm()
        normal_vector = normal_vector.subs({t: t_val}).evalf()
        return normal_vector

    def curve_tangent(self, t_val):
        t = sp.symbols('t')
        f = sp.Matrix([
            t,
            0.1*t**2,
            0
        ])

    def construct(self):
        self.camera.frame.set_euler_angles(phi=90*DEGREES, theta=180*DEGREES)
        self.camera.frame.set_height(32)

        # 2d axes
        axes_2d = Axes(x_range=[-3,3,1],
                       y_range=[-3.5,3.5,1]
                )
        axes_2d.move_to(ORIGIN + 10*RIGHT + 10*UP + 2*IN)
        axes_2d_origin = axes_2d.c2p(0,0)
        axes_2d.scale(1)
        axes_2d.rotate(PI/2, axis=RIGHT)
        # self.play(ShowCreation(axes_2d))


        # construct a curve
        curve = ParametricCurve(
                lambda t: np.array([t, 0.1*t**2, 0]),
                t_range = [-8,8, 0.1],
                color=BLUE
        ).shift(axes_2d_origin)
        curve.rotate(PI/2, axis=RIGHT, about_point = axes_2d_origin)
        curve.scale(1)

        self.play(ShowCreation(curve))

        # self.wait(1)

        # construct 3d axes
        axes_3d = ThreeDAxes(x_range=[-np.pi,np.pi,1], y_range=[-np.pi,np.pi,1], z_range=[-2,2,1])
        axes_3d.move_to(ORIGIN + 10*LEFT)
        axes_3d_origin = axes_3d.get_origin()
        axes_3d.scale(1)
        # self.play(ShowCreation(axes_3d))

        # construct a surface
        surface = ParametricSurface(
                lambda u, v: np.array([
                    3*u,
                    3*v,
                    3*np.sin(np.sqrt((u**2) + (v**2)))
                ]),
                u_range = [0,2*np.pi],
                v_range = [0, 2*np.pi],
                resolution=(20,40)
        ).move_to(axes_3d_origin)

        surface.set_color(BLUE)
        surface.set_opacity(0.9)
        surface.rotate(-PI, axis=RIGHT)
        surface.rotate(PI/6, axis=LEFT)
        surface.scale(1)

        self.play(ShowCreation(surface))

        # create normal vector on the curve
        t_val = 0
        n_curve = self.curve_normal(t_val)
        n_curve = np.array(n_curve).astype(np.float64).flatten()

        curve_n_pt = self.eval_curve(t_val) + axes_2d_origin
        curve_pt = Dot(point=curve_n_pt)
        curve_pt.set_color(PURPLE)
        curve_pt.set_opacity(1)

        print(curve_n_pt)

        self.add(curve_pt)

        # curve_n_vec = Line(curve_n_pt, curve_n_pt + n_curve)
        # curve_n_vec.set_color(YELLOW)
        # curve_n_vec.set_stroke(width=5)
        # curve_n_vec.scale(1.5, about_point=curve_n_pt)
        # curve_n_vec.rotate(90*DEGREES, axis=RIGHT, about_point=curve_n_pt)
        # self.play(ShowCreation(curve_n_vec))
        curve_n_vec = Arrow3D(curve_n_pt, curve_n_pt + n_curve,
                              color=YELLOW, thickness=0.02,
                              tip_width_ratio=3, tip_length=0.25
        )
        curve_n_vec.rotate(90*DEGREES, axis=RIGHT, about_point=curve_n_pt)
        self.play(ShowCreation(curve_n_vec))

        # curve normal vector label





        self.wait()
        self.embed()
