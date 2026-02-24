import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
import numpy as np
import sympy as sp
from Arrow3d import Arrow3d

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
        f_prime = f.diff(t)
        s_prime = sp.sqrt(f_prime.dot(f_prime))

        tangent_vector = f_prime/s_prime
        tangent_vector = tangent_vector/tangent_vector.norm()
        tangent_vector = tangent_vector.subs({t: t_val}).evalf()
        return tangent_vector

    def construct(self):
        self.camera.frame.set_euler_angles(phi=90*DEGREES, theta=180*DEGREES)
        self.camera.frame.set_height(32)

        # 2d axes
        axes_2d = Axes(x_range=[-3,3,1],
                       y_range=[-3.5,3.5,1]
                )
        axes_2d.move_to(ORIGIN + 10*RIGHT + 10*UP + 4*IN)
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

        original = self.eval_curve(t_val)
        rotated = np.array([original[0], -original[2], original[1]])
        curve_n_pt = rotated + axes_2d_origin
        curve_pt = Dot(point=curve_n_pt)
        curve_pt.set_color(PURPLE)
        curve_pt.set_opacity(1)

        print(curve_n_pt)

        self.add(curve_pt)

        curve_n_vec = Arrow3d(start=curve_n_pt,
                              end = curve_n_pt + n_curve,
                              thickness=0.15,
                              tip_radius=0.15,
                              color=YELLOW)
        curve_n_vec.rotate(90*DEGREES, axis=RIGHT, about_point=curve_n_pt)
        # curve_n_vec.set_opacity(0.75)
        self.play(ShowCreation(curve_n_vec))

        # create the tangent vector for the curve
        t_curve = self.curve_tangent(t_val) 
        t_curve = np.array(t_curve).astype(np.float64).flatten()

        curve_t_vec = Arrow3d(start=curve_n_pt,
                              end = curve_n_pt + t_curve,
                              thickness=0.15,
                              tip_radius=0.15,
                              color=YELLOW)
        # curve_t_vec.rotate(90*DEGREES, axis=LEFT, about_point=curve_n_pt)
        curve_t_vec.rotate(180*DEGREES, about_point=curve_n_pt)
        curve_t_vec.set_opacity(0.7)
        self.play(ShowCreation(curve_t_vec))


        # curve normal vector label
        curve_normal_label = Tex(r"\vec{n}").move_to(curve_n_pt + 1.5*OUT)
        curve_normal_label.set_color(YELLOW)
        curve_normal_label.rotate(90*DEGREES, axis=RIGHT)
        curve_normal_label.rotate(180*DEGREES)
        curve_normal_label.scale(1.75)

        # curve tangent vector label
        curve_tangent_label = Tex(r"\vec{t}").move_to(curve_n_pt + 1.5*LEFT + 0.5*IN)
        curve_tangent_label.set_color(YELLOW)
        curve_tangent_label.rotate(90*DEGREES, axis=RIGHT)
        curve_tangent_label.rotate(180*DEGREES)
        curve_tangent_label.scale(1.75)
        # curve_tangent_label.set_opacity(0.7)
        self.play(Write(curve_normal_label), Write(curve_tangent_label))


        # create the surface normal vector







        self.wait()
        self.embed()
