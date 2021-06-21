"""Test file to set up mixed SUPG problem for flow past cylinder"""
import logging
from os import path, remove
from shutil import copy, move
from time import ctime, time
from firedrake import VectorFunctionSpace, FunctionSpace, MixedFunctionSpace, \
    Function, TestFunction, TrialFunction, Constant, FacetNormal, sym, div, \
    nabla_grad, Identity, inner, dot, ds, lhs, rhs, dx, DirichletBC, grad, \
    LinearVariationalProblem, LinearVariationalSolver, File, COMM_WORLD, \
    sqrt, Dx, SpatialCoordinate, as_vector, conditional, assemble, perp, \
    DumbCheckpoint, FILE_READ, FILE_CREATE
from ufl import H1
from build_mesh import mesh_builder
#pylint: disable=invalid-name

SUPG = True
kappa_split = True
kappa_front = 1.
kappa_back = 1/6.
tau_coeff = 1/2.

solvers_verbose = False
maxk = 2
atol_hypre = 1e-8
rtol_hypre = 1e-5

pickup = True
dumpfreq = 100
dirname = 'run_2cm'
dirname_verbose = False
compressed_output = True
load_mesh = True
out_names = ['u_from_psi', 'norm_grad_u_from_psi']
# Can choose out names from
# 'velocity', 'pressure', 'grad_p', 'norm_grad_u'
# 'div_u', 'vort', 'rel_vort', 'cfl_nr', 'tau',
# 'stream', 'u_from_psi', 'norm_grad_u_from_psi'
# 'rel_stream', 'u_from_rel_psi', 'norm_grad_u_from_rel_psi'

test_past_cyl = False
if test_past_cyl:
    dx_ = 0.025
    dt = 0.002
    tmax = 1.5

    u0 = 1.5
    mu = 0.0001
    rho = 1

    Lx, Ly, cx, cy, cr = 1.2, 0.41, 0.2, 0.2, 0.05

    mesh_name = 'cylinder_flow'
    mbuilder = mesh_builder(load_mesh, mesh_name, dx_,
                            Lx=Lx, Ly=Ly, cx=cx, cy=cy, cr=cr)
else:
    dx_ = 0.01
    dt = 0.05
    tmax = 600*5

    u0 = 0.02
    mu = 0.001306
    rho = 999.7

    def Ly_b(Lxf, Lyf_f, slp_):
        """Set Ly_b according to outer trapezoid shape"""
        return Lyf_f + 2*Lxf*slp_

    dx_ = 0.5
    # Set up domain and mesh regions
    Lx, Ly = 30, 15
    cx, cr = 3, 0.15
    cy = 0.5*Ly + cr

    dxf3 = 0.1
    Lxf3, Lyf_f3, Lyf_b3, head_f3 = 20.9, 3.5, 10., 0.9
    fine_3 = (dxf3, head_f3, Lxf3, Lyf_f3, Lyf_b3)

    slope = (Lyf_b3 - Lyf_f3)/2./Lxf3

    dxf2 = 0.05
    Lxf2, Lyf_f2, head_f2 = 10.7, 2., 0.7
    fine_2 = (dxf2, head_f2, Lxf2, Lyf_f2, Ly_b(Lxf2, Lyf_f2, slope))

    dxf1 = 0.025
    Lxf1, Lyf_f1, head_f1 = 5.5, 1.2, 0.5
    fine_1 = (dxf1, head_f1, Lxf1, Lyf_f1, Ly_b(Lxf1, Lyf_f1, slope))

    dxf0 = 0.01
    Lxf0, Lyf_f0, head_f0 = 0.75, 0.5, 0.3
    fine_0 = (dxf0, head_f0, Lxf0, Lyf_f0, Ly_b(Lxf0, Lyf_f0, slope))

    dx_circ = 0.01
    dx_circ_ext = 1*dx_circ

    fine_md = (fine_3, fine_2, fine_1, fine_0)

    mesh_name = 'cylinder_flow'
    mbuilder = mesh_builder(load_mesh, mesh_name, dx_,
                            Lx=Lx, Ly=Ly, cx=cx, cy=cy, cr=cr,
                            dx_circ=dx_circ, dx_circ_ext=dx_circ_ext,
                            fine_mesh_data=fine_md)

mesh = mbuilder.get_mesh()

# Function spaces
DG0 = FunctionSpace(mesh, "DG", 0)
CG2 = FunctionSpace(mesh, "CG", 2)
Vu = VectorFunctionSpace(mesh, "CG", 2)
Vp = FunctionSpace(mesh, "CG", 1)
M = MixedFunctionSpace((Vu, Vp))

# Boundary conditions
x = SpatialCoordinate(mesh)
u0_diff = 0.005
u_wall = u0 - u0_diff
u_in = u0 - u_wall
u_in_expr = u_in*4*x[1]*(Ly - x[1])/Ly**2 + u_wall
u_in_func = Function(Vu)
u_in_func.project(as_vector([u_in_expr, 0.]))
p_in = (mu/rho)*8*u_in/Ly**2

t_indx = mbuilder.bdry_idcs['Wall_top'][0]
b_indx = mbuilder.bdry_idcs['Wall_bottom'][0]
in_indx = mbuilder.bdry_idcs['Inflow'][0]
out_indx = mbuilder.bdry_idcs['Outflow'][0]
c_indx = mbuilder.bdry_idcs['Circle'][0]

u_bcs_wall = DirichletBC(Vu, Constant((u_wall, 0.)), [t_indx, b_indx])
u_bcs_cyl = DirichletBC(Vu, Constant((0., 0.)), c_indx)
u_bcs_in = DirichletBC(Vu, u_in_func, in_indx)
p_bcs_out = DirichletBC(Vp, Constant(0.), out_indx)

u_bcs = [u_bcs_wall, u_bcs_in, u_bcs_cyl]
p_bcs = p_bcs_out

# Fields
xn = Function(M, name='xn')
un, pn = xn.split()
un.rename('u')
pn.rename('p')

xnp1 = Function(M)
unp1, pnp1 = xnp1.split()
ubar = 0.5*(un + unp1)
ustar = Function(Vu, name='ustar')
ustarbar = 0.5*(un + ustar)
if solvers_verbose:
    ustar_old = Function(Vu)

# Output fields
outfields = []
if 'velocity' in out_names:
    outfields.append(un)
if 'pressure' in out_names:
    outfields.append(pn)
if 'norm_grad_u' in out_names:
    norm_grad_u = Function(Vp, name='norm_grad_u')
    outfields.append(norm_grad_u)
if 'div_u' in out_names:
    div_u = Function(Vp, name='div_u')
    outfields.append(div_u)
if 'cfl_nr' in out_names:
    cfl_nr = Function(DG0, name='cfl number')
    outfields.append(cfl_nr)
if 'grad_p' in out_names:
    grad_p = Function(Vu, name='grad_p')
    outfields.append(grad_p)
if 'rel_vort' in out_names:
    u_b = Function(Vu)
    u_b.project(as_vector([u_in_expr, 0.]))
    rel_vort = Function(Vp, name='relative vorticity')
    outfields.append(rel_vort)
if 'vort' in out_names:
    vort = Function(Vp, name='vorticity')
    outfields.append(vort)
if ('stream' in out_names or 'rel_stream' in out_names
    or 'u_from_psi' in out_names):
    if 'vort' not in out_names:
        vort = Function(Vp)
        out_names.append('vort_stream')
    psi = Function(CG2, name='stream function')
    if 'stream' in out_names:
        outfields.append(psi)
    else:
        out_names.append('stream')
    if 'u_from_psi' in out_names:
        u_from_psi = Function(Vu, name='u from stream function')
        outfields.append(u_from_psi)
        if 'norm_grad_u_from_psi' in out_names:
            norm_grad_u_from_psi = Function(Vp, \
                name='norm_grad_u from stream function')
            outfields.append(norm_grad_u_from_psi)
if 'rel_stream' in out_names or 'u_from_rel_psi' in out_names:
    rel_psi = Function(CG2, name='relative stream function')
    if 'rel_stream' in out_names:
        outfields.append(rel_psi)
    else:
        out_names.append('rel_stream')
    if 'u_from_rel_psi' in out_names:
        u_from_rel_psi = Function(Vu, name='u from relative stream function')
        outfields.append(u_from_rel_psi)
        if 'norm_grad_u_from_rel_psi' in out_names:
            norm_grad_u_from_rel_psi = Function(Vp, \
                name='norm_grad_u from relative stream function')
            outfields.append(norm_grad_u_from_rel_psi)

# Initial conditions
un.project(as_vector([u_in_expr, 0.]))
pn.project(p_in*(Lx - x[0])/Lx)

# Directory and file names
if dirname_verbose:
    postfix = 'pred_corr'
    dir_name = "Temp_out/{0}_{1}".format(dirname, postfix)
    if SUPG:
        dir_name += "_SUPG"
else:
    dir_name = "Temp_out/{0}".format(dirname)
fname = dir_name + "/flow.pvd"

# Optionally load from h5 checkpoint file
if pickup:
    with DumbCheckpoint("{0}/chkpt_flow".format(dir_name),
                        mode=FILE_READ) as chk:
        chk.load(xn)
        init_t = chk.read_attribute("/", "time")
else:
    chkpt = DumbCheckpoint("{0}/chkpt_flow".format(dir_name),
                           mode=FILE_CREATE)
    init_t = 0.

# Constants and mesh related objects
test = TestFunction(DG0)
area = Function(DG0)
assemble(test*dx, tensor=area)
h_ = sqrt(area)

n = FacetNormal(mesh)
dt_ = Constant(dt)
nu = Constant(mu/rho)

# Functions to appear in form
def epsilon(uu):
    """Symmetric gradient"""
    return sym(nabla_grad(uu))

def sigma(uu, pp):
    """Stress tensor"""
    return 2*nu*epsilon(uu) - pp*Identity(len(uu))

# SUPG related terms
if SUPG:
    def res_u_func(u_adv, uu, pp):
        """Return expression for velocity equation residual"""
        uubar, ppbar = 0.5*(uu + un), 0.5*(pp + pn)
        return ((uu - un)/dt_ + dot(u_adv, nabla_grad(uubar))
                - div(sigma(uubar, ppbar)))

    if 'tau' in out_names:
        tau_out = Function(DG0, name='tau')
        outfields.append(tau_out)

    tau_coeff_ = Constant(tau_coeff)
    t_inv_dt = Constant(2.*kappa_front)/dt_
    t_inv_u = 2.*sqrt(dot(ubar, ubar))/h_
    t_inv_nu = 4.*nu/(h_**2)
    tau_expr = sqrt(t_inv_dt**2 + t_inv_u**2 + 9*t_inv_nu**2)
    if kappa_split:
        # Use less diffusive tau near cylinder
        sep = Constant(cx + 5*cr)
        tau = tau_coeff_*conditional(x[0] < sep, 1., 0.)/tau_expr
        t_inv_dt_back = Constant(2.*kappa_back)/dt_
        tau_expr_back = sqrt(t_inv_dt_back**2 + t_inv_u**2 + 9*t_inv_nu**2)
        tau += tau_coeff_*conditional(x[0] < sep, 0., 1.)/tau_expr_back
    else:
        tau = tau_coeff_/tau_expr

# Solver parameters
prms_fgmres = {'ksp_type': 'fgmres',
               'mat_type' : 'aij',
               'pc_type': 'gamg',
               'pc_gamg_sym_graph': True,
               'mg_levels': {'ksp_type': 'gmres',
                             'ksp_max_it': 5,
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}}

prms_gmres_hypre = {'ksp_type': 'gmres',
                    'ksp_atol': atol_hypre,
                    'ksp_rtol': rtol_hypre,
                    'pc_type': 'hypre',
                    'pc_hypre_type': 'boomeramg'}

prms_cg = {'ksp_type': 'cg',
           'ksp_rtol': 1e-6,
           'pc_type': 'bjacobi',
           'sub_pc_type': 'ilu'}

# Predictor
u_ = TrialFunction(Vu)
w = TestFunction(Vu)
u_bar = 0.5*(un + u_)

pred_eqn = (inner(w, (u_ - un)/dt_)*dx
            + inner(w, dot(ustarbar, nabla_grad(u_bar)))*dx
            + inner(sigma(u_bar, pn), epsilon(w))*dx
            - dot(w, dot(n, sigma(u_bar, pn)))*ds)

if SUPG:
    res_u_pred = res_u_func(ustarbar, u_, pn)
    pred_eqn += tau*inner(dot(ustarbar, nabla_grad(w)), res_u_pred)*dx

pred_p = LinearVariationalProblem(lhs(pred_eqn), rhs(pred_eqn),
                                  ustar, bcs=u_bcs)
pred_solver = LinearVariationalSolver(pred_p,
                                      solver_parameters=prms_gmres_hypre)

# Update
p_ = TrialFunction(Vp)
q = TestFunction(Vp)

upd_eqn = dt_*inner(grad(p_ - pn), grad(q))*dx + div(ustar)*q*dx
upd_p = LinearVariationalProblem(lhs(upd_eqn), rhs(upd_eqn),
                                 pnp1, bcs=p_bcs)
upd_solver = LinearVariationalSolver(upd_p, solver_parameters=prms_fgmres)

# Corrector
corr_eqn = inner(u_ - ustar, w)*dx + dt_*inner(grad(pnp1 - pn), w)*dx
u_corr_p = LinearVariationalProblem(lhs(corr_eqn), rhs(corr_eqn), unp1)
corr_solver = LinearVariationalSolver(u_corr_p, solver_parameters=prms_cg)

# Build output field expressions and pvd file
if ('norm_grad_u' in out_names or 'norm_grad_u_from_rel_psi' in out_names
        or 'norm_grad_u_from_psi' in out_names):
    n_grd_u_expr = lambda uu: \
        sqrt(inner(Dx(uu, 0), Dx(uu, 0)) + inner(Dx(uu, 1), Dx(uu, 1)))
if 'cfl_nr' in out_names:
    cfl_expr = sqrt(dot(un, un))*dt_/h_
if 'vort' in out_names or 'vort_stream' in out_names:
    vort_expr = -div(perp(un))
if 'rel_vort' in out_names:
    rel_vort_expr = -div(perp(un - u_b))
if 'stream' in out_names or 'rel_stream' in out_names:
    psi_ = TrialFunction(CG2)
    phi = TestFunction(CG2)
    psi_eqn = inner(grad(phi), grad(psi_))*dx - phi*vort*dx

    # Psi bc in accordance with u_in_expr
    y_int_const = - Ly*(u_in/3. + u_wall/2.)
    psi_in = u_in*4*x[1]**2*(Ly/2. - x[1]/3.)/Ly**2 + x[1]*u_wall + y_int_const
    psi_b = y_int_const
    psi_t = - y_int_const

    psi_bcs = [DirichletBC(CG2, psi_t, t_indx),
               DirichletBC(CG2, psi_b, b_indx),
               DirichletBC(CG2, psi_in, in_indx)]

    psi_bcs.append(DirichletBC(CG2, Constant(0.), c_indx))
    psi_p = LinearVariationalProblem(lhs(psi_eqn), rhs(psi_eqn),
                                     psi, bcs=psi_bcs)
    psi_solver = LinearVariationalSolver(psi_p, solver_parameters=prms_cg)

    # Psi-dependent diagnostic setup
    u_from_psi_expr = lambda cc: - perp(grad(cc))
    if 'rel_stream' in out_names:
        psi_b = Function(CG2, name='relative psi')
        psi_b.project(psi_in)

mode = "a" if pickup else "w"
if compressed_output:
    outfile = File(fname, mode=mode, target_degree=1, target_continuity=H1)
else:
    outfile = File(fname, mode=mode)

# Also copy code for output version control
if COMM_WORLD.Get_rank() == 0:
    copy('Incompressible_SUPG.py', dir_name)

    copy('build_mesh.py', dir_name)
    geo_name = '{0}.geo'.format(mesh_name)
    msh_name = '{0}.msh'.format(mesh_name)
    if path.isfile(dir_name + '/' + geo_name):
        remove(dir_name + '/' + geo_name)
    if path.isfile(dir_name + '/' + msh_name):
        remove(dir_name + '/' + msh_name)
    if load_mesh:
        copy(geo_name, dir_name)
        copy(msh_name, dir_name)
    else:
        move(geo_name, dir_name)
        move(msh_name, dir_name)

def write(c_, t_):
    """Function to compute diagnostic fields and write output"""
    if c_ % dumpfreq == 0:
        # Write to vtu file
        if 'norm_grad_u' in out_names:
            norm_grad_u.project(n_grd_u_expr(un))
        if 'div_u' in out_names:
            div_u.project(div(un))
        if 'cfl_nr' in out_names:
            cfl_nr.project(cfl_expr)
        if 'grad_p' in out_names:
            grad_p.project(grad(pn))
        if 'vort' in out_names or 'vort_stream' in out_names:
            vort.project(vort_expr)
        if 'rel_vort' in out_names or 'rel_vort_stream' in out_names:
            rel_vort.project(rel_vort_expr)
        if 'stream' in out_names or 'rel_stream' in out_names:
            psi_solver.solve()
            if 'rel_stream' in out_names:
                rel_psi.project(psi - psi_b)
            if 'u_from_psi' in out_names:
                u_from_psi.project(u_from_psi_expr(psi))
                if 'norm_grad_u_from_psi' in out_names:
                    norm_grad_u_from_psi.project(n_grd_u_expr(u_from_psi))
            if 'u_from_rel_psi' in out_names:
                u_from_rel_psi.project(u_from_psi_expr(rel_psi))
                if 'norm_grad_u_from_rel_psi' in out_names:
                    norm_grad_u_from_rel_psi.project( \
                        n_grd_u_expr(u_from_rel_psi))
        if SUPG and 'tau' in out_names:
            tau_out.project(tau)

        outfile.write(*outfields)

        # Write to h5 file
        chkpt.store(xn)
        chkpt.write_attribute("/", "time", t_)

if not pickup:
    write(0, init_t)

# Logger
logger = logging.getLogger("Navier Stokes SUPG")

def set_log_handler(comm):
    """Set up logger handler for first parallel rank"""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(\
        fmt="%(name)s:%(levelname)s %(message)s"))
    if logger.hasHandlers():
        logger.handlers.clear()
    if comm.rank == 0:
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())


set_log_handler(mesh.comm)
logger.setLevel('INFO')

# Time loop
ustar.assign(un)

t = init_t
count = 0 if not pickup else int(t/dt + 1e-9)
while t < tmax + 0.5*dt:
    logger.info("Time step nr {0} at {1}; t = {2}"\
                .format(count, ctime(), round(t, 4)))

    t += dt

    # Run solvers
    ustar.assign(un)
    time1 = time()
    for _ in range(maxk):
        if solvers_verbose:
            ustar_old.assign(ustar)
        pred_solver.solve()
        if solvers_verbose:
            logger.info("Ustar residual in predictor loop: {0}"
                        .format( sqrt(assemble(inner(ustar - ustar_old,
                                                     ustar - ustar_old)*dx))))

    time2 = time()
    upd_solver.solve()
    time3 = time()
    corr_solver.solve()
    time4 = time()

    if solvers_verbose:
        logger.info("Pre solve time time: {0} sec" \
                    .format(time2 - time1))
        logger.info("Upd solve time time: {0} sec" \
                    .format(time3 - time2))
        logger.info("Cor solve time time: {0} sec" \
                    .format(time4 - time3))

    xn.assign(xnp1)

    # Write output
    count += 1
    write(count, t)
