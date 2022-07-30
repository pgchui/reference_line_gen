#include <OsqpEigen/OsqpEigen.h>

int main(int argc, char** argv)
{
    // declare QP variables
    size_t num_variables, num_constraints;
    Eigen::SparseMatrix<double> mat_q;
    Eigen::VectorXd vec_p;
    Eigen::SparseMatrix<double> mat_linear_constraints;
    Eigen::VectorXd vec_lower_bounds, vec_upper_bounds;
    // declare QP solver
    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.settings()->setVerbosity(true);
    // init QP solver
    solver.data()->setNumberOfVariables(num_variables);
    solver.data()->setNumberOfConstraints(num_constraints);
    if (!solver.data()->setHessianMatrix(mat_q)) return EXIT_FAILURE;
    if (!solver.data()->setGradient(vec_p))  return EXIT_FAILURE;
    if (!solver.data()->setLinearConstraintsMatrix(mat_linear_constraints))   return EXIT_FAILURE;
    if (!solver.data()->setLowerBound(vec_lower_bounds))    return EXIT_FAILURE;
    if (!solver.data()->setUpperBound(vec_upper_bounds))    return EXIT_FAILURE;
    if (!solver.initSolver())   return EXIT_FAILURE;
    // solve QP problem
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return EXIT_FAILURE;
    auto qp_solution = solver.getSolution();
    // update QP constraints
    if (!solver.updateBounds(vec_lower_bounds, vec_upper_bounds)) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}