"""File to create .msh file given .geo file and a specified resolution."""
import subprocess
from firedrake import Mesh, COMM_WORLD

__all__ = ["mesh_builder"]


class mesh_builder():
    """Class to build geo files and corresponding mesh for Navier Stokes
    channel flow test cases.
    :arg load_mesh: load mesh instead of building it. Used in environments
    where mesh can't be build. To ensure correct boundary condition indeces,
    the mesh file that is loaded should have been built using the same script
    as created in the mesh_builder object.
    :arg mesh_name: name of mesh file to be created
    :arg dx: average cell side length, defaults to 0.05
    :args Lx, Ly: dimensions of channel; default to 1.
    :arg fine_res_bdry: switch on whether to include a finer mesh region along
    the domain's outer boundary; defaults to True.
    :arg include_cylinder: switch on whether to include a cylinder in mesh;
    defaults to True.
    :args cx, cy, cr: cylinder centre x and y coordinates and radius; default
    to Lx/6., Ly/2., and Lx/20.
    :arg dx_circ: average cell side length near cylinder, defaults to dx
    (or dxf if a fine resolution is incldued)
    :arg dx_circ_ext: extend away from cylinder in which to use dx_circ
    resolution; defaults to None (in which case resolution adjusts to outer
    resolution as set up in mesher)
    :arg fine_mesh_data: list containing 5-tuples for fine mesh trapezoids around
    cylinder of form (fine mesh resolution, distance from rectangle's left edge to
    circle, centre fine mesh trapezoidal domain's height and lengths)
    cylinder; defaults to None. List of tuples assumes trapezoids inwards, i.e.
    each one should contain the next one, with the last one containing the circle.
    """
    def __init__(self, load_mesh, mesh_name, dx=0.05, Lx=1, Ly=1,
                 fine_res_bdry=False, include_cylinder=True, cx=None, cy=None,
                 cr=None, dx_circ=None, dx_circ_ext=None, fine_mesh_data=None):

        self.mesh_data = [dx, Lx, Ly,
                          Lx/6. if cx is None else cx,
                          Ly/2. if cy is None else cy,
                          Lx/20. if cr is None else cr]

        if dx_circ is None:
            if fine_mesh_data is not None:
                dx_circ = fine_mesh_data[-1][0]
            else:
                dx_circ = dx
        self.mesh_data.append(dx_circ)

        # Build .geo and .msh files
        self.mname_mesh = mesh_name + '.msh'
        self.mname_geo = mesh_name + '.geo'
        # Build script on all ranks for boundary indeces
        script = self._build_geo_script(include_cylinder, fine_res_bdry,
                                        dx_circ_ext, fine_mesh_data)
        # Build file on one rank only
        if COMM_WORLD.Get_rank() == 0 and not load_mesh:
            with open(self.mname_geo, 'w') as geo_file:
                geo_file.write(script)
            bashCommand = 'gmsh -2 {0} -format msh2'.format(self.mname_geo)
            process = subprocess.Popen(bashCommand.split(),
                                       stdout=subprocess.PIPE)
            process.communicate()

    def get_mesh(self):
        """Load and return mesh from .msh file"""
        # Return mesh loaded from .msh file
        return Mesh(self.mname_mesh)

    def _build_geo_script(self, cyl, bdry, circ_ext, f_msh_dat):
        """Create .geo file script for channel flow according
        to specifications."""
        dx, Lx, Ly, cx, cy, cr, dx_circ = self.mesh_data
        fine_mesh_data = f_msh_dat
        self.p_count = 0
        self.l_count = 0
        self.s_count = 0
        self.line_loops_idcs = {}
        self.bdry_idcs = {'Circle': None, 'Wall_top': None, 'Wall_bottom': None,
                          'Inflow': None, 'Outflow': None}
        script = '// Channel mesh with immersed circle;\n'
        loop_names = []

        # Set up domain rectangle
        name = 'rectangle'
        if bdry:
            bdry_dx = dx/8.
        else:
            bdry_dx = dx
        phys_dict_r = {'Wall_top': ['top', ], 'Wall_bottom': ['bottom', ],
                       'Inflow': ['left', ], 'Outflow': ['right', ]}
        script += self._rectangle(0, Lx, 0, Ly, bdry_dx, name, phys_dict_r)
        loop_names.append(name)

        # Set up fine rectangle near domain boundary
        if bdry:
            name = 'rectangle_bdry'
            bdry_width = 2*dx
            script += self._rectangle(bdry_width, Lx - bdry_width,
                                      bdry_width, Ly - bdry_width, dx, name)
            loop_names.append(name)

        # Set up fine trapezoids around cylinder
        if fine_mesh_data is not None:
            for idx, params in enumerate(fine_mesh_data):
                name = 'fine_trapezoid_{0}'.format(idx + 1)
                dxf, head_f, Lxf, Lyf_f, Lyf_b = params
                script += self._rectangle(cx - head_f, cx - head_f + Lxf,
                                          cy - Lyf_f/2., cy + Lyf_f/2., dxf, name,
                                          Ly_bl=cy - Lyf_b/2.,
                                          Ly_bu=cy + Lyf_b/2.)
                loop_names.append(name)


        # Set up dx_circ region cylinder
        if circ_ext is not None and cyl:
            name = 'circle_ext'
            script += self._cylinder(cx, cy, cr + circ_ext,
                                     dx_circ, name)
            loop_names.append(name)

        # Set up cylinder
        if cyl:
            name = 'circle'
            script += self._cylinder(cx, cy, cr, dx_circ, name,
                                     physical=True)
            loop_names.append(name)


        # Set up surfaces depending on resolution subdivisions
        # Script for one region
        if len(loop_names) == 1:
            script += 'Plane Surface(1) = {{{0}}};\n'\
                      .format(self.line_loops_idcs['rectangle'])
            script += 'Physical Surface("Domain", 2) = {1};'
            return script

        # Script for several regions
        s_ = 1
        for i in range(len(loop_names) - 1):
            script += 'Plane Surface({2}) = {{{0}, {1}}};\n'\
                      .format(self.line_loops_idcs[loop_names[i]],
                              self.line_loops_idcs[loop_names[i + 1]], s_)
            script += 'Physical Surface("Domain_{0}", {1}) = {0};\n'\
                      .format(s_, s_ + 1)
            s_ += 1

        # If there is no cylinder but more than one region make sure inner
        # region also contains mesh
        if not cyl:
            script += 'Plane Surface({1}) = {{{0}}};\n'\
                      .format(self.line_loops_idcs[loop_names[-1]], s_ + 1)
            script += 'Physical Surface("Domain_{0}", {1}) = {2};'\
                      .format(s_, s_ + 2, s_ + 1)
        return script


    def _rectangle(self, x0, x1, y0, y1, dx, loop_identifier, phys_dict={},
                   Ly_bl=None, Ly_bu=None):
        """Build rectangle for .geo file, numbering points and lines
        accordingly. Create physical lines according to phys_dict.
        Optionally pass the y-coordinates of the lower and upper vertices
        of the rectangle's right side, so that instead a trapezoid is returned.
        """
        if ((Ly_bl is None and Ly_bu is not None)
                or (Ly_bl is not None and Ly_bu is None)):
            raise RuntimeError('If specifying right edge vertex locations, '\
                               'must specify both of them.')
        if Ly_bl is None and Ly_bu is None:
            y2, y3 = y0, y1
        else:
            y2 = Ly_bl
            y3 = Ly_bu

        pc, lc = self.p_count, self.l_count

        # Build points
        rect_script = ('Point({7}) = {{{0}, {3}, 0, {6}}};\n'
                       'Point({8}) = {{{0}, {2}, 0, {6}}};\n'
                       'Point({9}) = {{{1}, {4}, 0, {6}}};\n'
                       'Point({10}) = {{{1}, {5}, 0, {6}}};\n')\
                       .format(x0, x1, y0, y1, y2, y3, dx,
                               pc + 1, pc + 2, pc + 3, pc + 4)

        # Build lines: top, right, bottom, left
        rect_script += ('Line({4}) = {{{0}, {3}}};\n'
                        'Line({5}) = {{{3}, {2}}};\n'
                        'Line({6}) = {{{2}, {1}}};\n'
                        'Line({7}) = {{{1}, {0}}};\n')\
                        .format(pc + 1, pc + 2, pc + 3, pc + 4,
                                lc + 1, lc + 2, lc + 3, lc + 4)

        # Build line loop
        rect_script += ('Line Loop({4}) = {{{0}, {1}, {2}, {3}}};\n'\
                        .format(lc + 1, lc + 2, lc + 3, lc + 4, lc + 5))
        self.line_loops_idcs[loop_identifier] = lc + 5

        # Build physical lines according to dictionary
        plc = 0
        line_dict = {'top': lc + 1, 'right': lc + 2,
                     'bottom': lc + 3, 'left': lc + 4}

        for name, sides in phys_dict.items():
            pl_script = 'Physical Line("{0}", {1}) = {{'.format(name,
                                                                lc + 6 + plc)
            # Add index to boundary indeces dictionary for DirichletBC
            if self.bdry_idcs[name] is None:
                self.bdry_idcs[name] = [lc + 6 + plc, ]
            else:
                self.bdry_idcs[name].append(lc + 6 + plc)
            for side in sides:
                pl_script += '{0}, '.format(line_dict[side])
            pl_script = pl_script[:-2]
            pl_script += '};\n'
            plc += 1
            rect_script += pl_script

        pc += 4
        lc += 5 + len(phys_dict)
        self.p_count, self.l_count = pc, lc

        return rect_script

    def _cylinder(self, cx, cy, cr, dx, loop_identifier, physical=False,
                  flow_res_fac=1):
        '''Build cylidner for .geo file, numbering points from pc. flow_res_fac
        scales the resolution in the direction of the flow (noting that the
        flow speed on the sides will be faster than upfront and at the back).'''
        pc, lc = self.p_count, self.l_count

        # Build points
        cyl_script = ('Point({5}) = {{{0},       {1},       0, {3}}};\n'
                      'Point({6}) = {{{0} + {2}, {1},       0, {4}}};\n'
                      'Point({7}) = {{{0} - {2}, {1},       0, {4}}};\n'
                      'Point({8}) = {{{0},       {1} + {2}, 0, {3}}};\n'
                      'Point({9}) = {{{0},       {1} - {2}, 0, {3}}};\n')\
                      .format(cx, cy, cr, dx, flow_res_fac*dx,
                              pc + 1, pc + 2, pc + 3, pc + 4, pc + 5)
        # Build segments
        cyl_script += ('Circle({5}) = {{{3}, {0}, {1}}};\n'
                       'Circle({6}) = {{{1}, {0}, {4}}};\n'
                       'Circle({7}) = {{{4}, {0}, {2}}};\n'
                       'Circle({8}) = {{{2}, {0}, {3}}};\n')\
                       .format(pc + 1, pc + 2, pc + 3, pc + 4, pc + 5,
                               lc + 1, lc + 2, lc + 3, lc + 4)
        # Build line loop
        cyl_script += 'Line Loop({4}) = {{{3}, {0}, {1}, {2}}};\n'\
                      .format(lc + 1, lc + 2, lc + 3, lc + 4, lc + 5)
        self.line_loops_idcs[loop_identifier] = lc + 5

        # Build physical line
        if physical:
            cyl_script += 'Physical Line("Circle", {4})'\
                          ' = {{{3}, {0}, {1}, {2}}};\n'\
                          .format(lc + 1, lc + 2, lc + 3, lc + 4, lc + 6)

            # Add index to boundary indeces dictionary for DirichletBC
            if self.bdry_idcs['Circle'] is None:
                self.bdry_idcs['Circle'] = [lc + 6, ]
            else:
                self.bdry_idcs['Circle'].append(lc + 6)
            lc_phys = 1
        else:
            lc_phys = 0

        pc += 5
        lc += 6 + lc_phys
        self.p_count, self.l_count = pc, lc
        return cyl_script
