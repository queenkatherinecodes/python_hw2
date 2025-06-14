import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
import io
from .examples import (qp, qp_eq_constraints_mat, qp_eq_constraints_rhs, qp_ineq_constraints, 
                      lp, lp_eq_constraints_mat, lp_eq_constraints_rhs, lp_ineq_constraints)
from src.constrained_min import interior_pt


class TestConstrainedMin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up PDF file for saving results - called once for the entire test class"""
        cls.pdf_filename = "constrained_optimization_results.pdf"
        cls.pdf_pages = PdfPages(cls.pdf_filename)
    
    @classmethod
    def tearDownClass(cls):
        """Close PDF file - called once after all tests are done"""
        cls.pdf_pages.close()
        print(f"\nResults saved to {cls.pdf_filename}")
    
    def add_text_page_to_pdf(self, title, text_content):
        """Add a text page to the PDF"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.1, 0.9, title, fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.1, text_content, fontsize=12, transform=ax.transAxes, 
                verticalalignment='bottom', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def test_qp(self):
        """Test quadratic programming example with 3D feasible region"""
        print("\n=== Quadratic Programming Test ===")
        
        x0 = np.array([0.1, 0.2, 0.7])
        x_k, f_x_k, history = interior_pt(qp, qp_ineq_constraints(), qp_eq_constraints_mat(), qp_eq_constraints_rhs(), x0)
        path = np.array([point[0] for point in history])
        objective_values = [point[1] for point in history]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'k-', linewidth=2, label='Feasible region boundary')

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        triangle = [[vertices[0], vertices[1], vertices[2]]]
        ax.add_collection3d(Poly3DCollection(triangle, alpha=0.3, facecolor='lightblue', edgecolor='black'))
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'ro-', markersize=6, linewidth=2, label='Central path')
        ax.plot([x_k[0]], [x_k[1]], [x_k[2]], 'gs', markersize=10, label=f'Final solution: ({x_k[0]:.4f}, {x_k[1]:.4f}, {x_k[2]:.4f})')
        ax.set_xlabel('x₀')
        ax.set_ylabel('x₁')
        ax.set_zlabel('x₂')
        ax.set_title('QP: Feasible Region and Central Path')
        ax.legend()
        self.pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(objective_values)), objective_values, 'bo-', markersize=6)
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('QP: Objective Value vs Iteration')
        ax.grid(True)
        self.pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        eq_constraint_value = x_k[0] + x_k[1] + x_k[2] - 1

        text_content = f"""QUADRATIC PROGRAMMING RESULTS

Final Solution:
x = [{x_k[0]:.6f}, {x_k[1]:.6f}, {x_k[2]:.6f}]

Objective Value:
f(x) = {f_x_k:.6f}

Constraint Values:
Equality constraint (x₀+x₁+x₂-1): {eq_constraint_value:.6f}
Inequality constraints:
  x₀ ≥ 0: {x_k[0]:.6f}
  x₁ ≥ 0: {x_k[1]:.6f}
  x₂ ≥ 0: {x_k[2]:.6f}

Number of outer iterations: {len(objective_values)}
"""

        self.add_text_page_to_pdf("QUADRATIC PROGRAMMING RESULTS", text_content)

        print(f"Final solution: x = [{x_k[0]:.6f}, {x_k[1]:.6f}, {x_k[2]:.6f}]")
        print(f"Final objective value: {f_x_k:.6f}")
        print(f"Equality constraint (x₀+x₁+x₂-1): {eq_constraint_value:.6f}")
        print(f"Inequality constraints (x₀, x₁, x₂): ({x_k[0]:.6f}, {x_k[1]:.6f}, {x_k[2]:.6f})")
    
    def test_lp(self):
        """Test linear programming example with 2D feasible region"""
        print("\n=== Linear Programming Test ===")
        
        x0 = np.array([0.5, 0.75])
        x_k, f_x_k, history = interior_pt(lp, lp_ineq_constraints(), lp_eq_constraints_mat(), lp_eq_constraints_rhs(), x0)
        path = np.array([point[0] for point in history])
        objective_values = [point[1] for point in history]
        fig, ax = plt.subplots(figsize=(10, 8))
        x = np.linspace(-0.5, 2.5, 300)
        y = np.linspace(-0.5, 1.5, 300)
        X, Y = np.meshgrid(x, y)
        constraint1 = Y >= X - 1  
        constraint2 = Y <= 1     
        constraint3 = X <= 2    
        constraint4 = Y >= 0
        feasible = constraint1 & constraint2 & constraint3 & constraint4
        ax.contourf(X, Y, feasible.astype(int), levels=[0.5, 1.5], colors=['lightblue'], alpha=0.5)
        ax.contour(X, Y, feasible.astype(int), levels=[0.5], colors=['black'], linewidths=2)
        ax.plot(x, x - 1, 'k--', label='y = x - 1', alpha=0.7)
        ax.axhline(y=1, color='k', linestyle='--', label='y = 1', alpha=0.7)
        ax.axvline(x=2, color='k', linestyle='--', label='x = 2', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', label='y = 0', alpha=0.7)
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=6, linewidth=2, label='Central path')
        ax.plot([x_k[0]], [x_k[1]], 'gs', markersize=10, label=f'Final solution: ({x_k[0]:.4f}, {x_k[1]:.4f})')
        ax.set_xlabel('x₀')
        ax.set_ylabel('x₁')
        ax.set_title('LP: Feasible Region and Central Path')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 2.5)
        self.pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(objective_values)), objective_values, 'bo-', markersize=6)
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('LP: Objective Value vs Iteration')
        ax.grid(True)
        self.pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        constraint_values = []
        constraint_names = ['y - (x-1)', '1 - y', '2 - x', 'y']
        constraint_values.append(x_k[1] - x_k[0] + 1)
        constraint_values.append(1 - x_k[1])
        constraint_values.append(2 - x_k[0])
        constraint_values.append(x_k[1])
        constraint_text = "\n".join([f"  {name}: {value:.6f}" 
                                   for name, value in zip(constraint_names, constraint_values)])

        text_content = f"""LINEAR PROGRAMMING RESULTS

Final Solution:
x = [{x_k[0]:.6f}, {x_k[1]:.6f}]

Objective Value:
f(x) = {f_x_k:.6f}
Original objective (-(x₀+x₁)): {-(x_k[0] + x_k[1]):.6f}

Constraint Values (should be ≥ 0):
{constraint_text}

Number of outer iterations: {len(objective_values)}
"""
        self.add_text_page_to_pdf("LINEAR PROGRAMMING RESULTS", text_content)
        
        print(f"Final solution: x = [{x_k[0]:.6f}, {x_k[1]:.6f}]")
        print(f"Final objective value: {f_x_k:.6f}")
        print(f"Original objective (-(x₀+x₁)): {-(x_k[0] + x_k[1]):.6f}")
        print("Inequality constraint values (should be >= 0):")
        for name, value in zip(constraint_names, constraint_values):
            print(f"  {name}: {value:.6f}")


if __name__ == '__main__':
    unittest.main()