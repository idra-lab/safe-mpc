import re

equations = '''
fExpl = {
            (-params.g * params.l1 * params.l2 * params.l2 * params.m1 * params.m3 * sin(
                -2 * self.x[2] + 2 * self.x[1] + self.x[
                    0]) - params.g * params.l1 * params.l2 * params.l2 * params.m1 * params.m3 * sin(
                2 * self.x[2] - 2 * self.x[1] + self.x[0]) + 2 * self.u[0] * params.l2 * params.l2 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) + 2 * self.x[
                 3] ** 2 * params.l1 ** 2 * params.l2 * params.l2 * params.m2 * (
                     params.m2 + params.m3) * sin(-2 * self.x[1] + 2 * self.x[0]) - 2 * self.u[
                 2] * params.l1 * params.l2 * (
                     params.m2 + params.m3) * cos(
                -2 * self.x[1] + self.x[0] + self.x[2]) - 2 * self.u[1] * params.l1 * params.l2 * params.m3 * cos(
                -2 * self.x[2] + self.x[1] + self.x[
                    0]) + 2 * params.l1 * params.l2 * params.l2 ** 2 * params.m2 * params.m3 * self.x[5] ** 2 * sin(
                -2 * self.x[1] + self.x[0] + self.x[2]) + 2 * self.u[2] * params.l1 * params.l2 * (
                     params.m2 + params.m3) *
             cos(self.x[0] - self.x[2]) + 2 * (
                     self.u[1] * params.l1 * (params.m3 + 2 * params.m2) * cos(-self.x[1] + self.x[0]) + (
                     params.g * params.l1 * params.m2 * (params.m2 + params.m3) * sin(
                 -2 * self.x[1] + self.x[0]) + 2 * self.x[4] ** 2 * params.l1 * params.l2 * params.m2 * (
                             params.m2 + params.m3) * sin(-self.x[1] + self.x[0]) + params.m3 * self.x[
                         5] ** 2 * sin(
                 self.x[0] - self.x[2]) * params.l1 * params.l2 * params.m2 + params.g * params.l1 * (
                             params.m2 ** 2 + (
                             params.m3 + 2 * params.m1) * params.m2 + params.m1 * params.m3) * sin(
                 self.x[0]) - self.u[0] * (
                             params.m3 + 2 * params.m2)) * params.l2) * params.l2) / params.l1 ** 2 / params.l2 / (
                    params.m2 * (params.m2 + params.m3) * cos(
                -2 * self.x[1] + 2 * self.x[0]) + params.m1 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) - params.m2 ** 2 + (
                            -params.m3 - 2 * params.m1) * params.m2 - params.m1 * params.m3) / params.l2 / 2 - self.a[0],
            (-2 * self.u[2] * params.l1 * params.l2 * (params.m2 + params.m3) * cos(
                2 * self.x[0] - self.x[2] - self.x[
                    1]) - 2 * params.l1 * params.l2 * params.l2 ** 2 * params.m2 * params.m3 * self.x[5] ** 2 * sin(
                2 * self.x[0] - self.x[2] - self.x[
                    1]) + params.g * params.l1 * params.l2 * params.l2 * params.m1 * params.m3 * sin(
                self.x[1] + 2 * self.x[0] - 2 * self.x[2]) - params.g * params.l1 * params.l2 * (
                     (params.m1 + 2 * params.m2) * params.m3 + 2 * params.m2 * (
                     params.m1 + params.m2)) * params.l2 * sin(
                -self.x[1] + 2 * self.x[0]) - 2 * self.x[
                 4] ** 2 * params.l1 * params.l2 ** 2 * params.l2 * params.m2 * (
                     params.m2 + params.m3) * sin(
                -2 * self.x[1] + 2 * self.x[0]) + 2 * self.u[1] * params.l1 * params.l2 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[0]) + 2 * params.l1 * params.l2 ** 2 * params.l2 * params.m1 * params.m3 *
             self.x[4] ** 2 * sin(
                        -2 * self.x[2] + 2 * self.x[1]) - 2 * self.u[0] * params.l2 * params.l2 * params.m3 * cos(
                        -2 * self.x[2] + self.x[1] + self.x[
                            0]) + 2 * params.l1 ** 2 * params.l2 * params.l2 * params.m1 * params.m3 * self.x[
                 3] ** 2 * sin(
                        -2 * self.x[2] +
                        self.x[1] + self.x[0]) - 2 * params.l1 ** 2 * params.l2 * self.x[3] ** 2 * (
                     (params.m1 + 2 * params.m2) * params.m3 + 2 * params.m2 * (
                     params.m1 + params.m2)) * params.l2 * sin(
                        -self.x[1] + self.x[0]) + 2 * self.u[2] * params.l1 * params.l2 * (
                     params.m3 + 2 * params.m1 + params.m2) * cos(
                        -self.x[2] + self.x[1]) + (2 * self.u[0] * params.l2 * (params.m3 + 2 * params.m2) * cos(
                        -self.x[1] + self.x[0]) + params.l1 * (
                                                           4 * self.x[5] ** 2 * params.m3 * params.l2 * (
                                                           params.m1 + params.m2 / 2) * params.l2 * sin(
                                                       -self.x[2] + self.x[
                                                           1]) + params.g * params.m3 * params.l2 * params.m1 * sin(
                                                       -2 * self.x[2] + self.x[1]) + params.g * (
                                                                   (
                                                                           params.m1 + 2 * params.m2) * params.m3 + 2 * params.m2 * (
                                                                           params.m1 + params.m2)) * params.l2 * sin(
                                                       self.x[1]) - 2 * self.u[1] * (
                                                                   params.m3 + 2 * params.m1 + 2 * params.m2))) * params.l2) / (
                    params.m2 * (params.m2 + params.m3) * cos(
                -2 * self.x[1] + 2 * self.x[0]) + params.m1 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) + (
                            -params.m1 - params.m2) * params.m3 - 2 * params.m1 * params.m2 - params.m2 ** 2) / params.l1 / params.l2 / params.l2 ** 2 / 2 - self.a[1],
            (-2 * params.m3 * self.u[1] * params.l1 * params.l2 * (params.m2 + params.m3) * cos(
                2 * self.x[0] - self.x[2] - self.x[
                    1]) + params.g * params.m3 * params.l1 * params.l2 * params.l2 * params.m1 * (
                     params.m2 + params.m3) * sin(2 * self.x[0] + self.x[2] - 2 * self.x[1]) + 2 * self.u[
                 2] * params.l1 * params.l2 * (
                     params.m2 + params.m3) ** 2 * cos(
                -2 * self.x[1] + 2 * self.x[
                    0]) - params.g * params.m3 * params.l1 * params.l2 * params.l2 * params.m1 * (
                     params.m2 + params.m3) * sin(
                2 * self.x[0] - self.x[2]) - params.g * params.m3 * params.l1 * params.l2 * params.l2 * params.m1 * (
                     params.m2 + params.m3) * sin(
                -self.x[2] + 2 * self.x[1]) - 2 * params.l1 * params.l2 * params.l2 ** 2 * params.m1 * params.m3 ** 2 *
             self.x[5] ** 2 * sin(
                        -2 * self.x[2] + 2 * self.x[1]) - 2 * self.u[0] * params.l2 * params.l2 * params.m3 * (
                     params.m2 + params.m3) * cos(
                        -2 * self.x[1] + self.x[0] + self.x[2]) + 2 * params.m3 * self.x[3] ** 2 * params.l1 ** 2 *
             params.l2 * params.l2 * params.m1 * (params.m2 + params.m3) * sin(
                        -2 * self.x[1] + self.x[0] + self.x[2]) + 2 * params.m3 * self.u[1] * params.l1 * params.l2 * (
                     params.m3 + 2 * params.m1 + params.m2) * cos(-self.x[2] + self.x[1]) + (params.m2 + params.m3) * (
                     2 * self.u[0] * params.l2 * params.m3 * cos(self.x[0] - self.x[2]) + params.l1 * (
                     -2 * params.m3 * self.x[3] ** 2 * params.l1 * params.l2 * params.m1 * sin(
                 self.x[0] - self.x[2]) - 4 * params.m3 * self.x[4] ** 2 * sin(
                 -self.x[2] + self.x[1]) * params.l2 * params.l2 * params.m1 + params.g * params.m3 * sin(
                 self.x[2]) * params.l2 * params.m1 - 2 * self.u[2] * (
                             params.m3 + 2 * params.m1 + params.m2))) * params.l2) / params.m3 / (
                    params.m2 * (params.m2 + params.m3) * cos(
                -2 * self.x[1] + 2 * self.x[0]) + params.m1 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) + (
                            -params.m1 - params.m2) * params.m3 - 2 * params.m1 * params.m2 - params.m2 ** 2) / params.l1 / params.l2 ** 2 / params.l2 / 2 - self.a[2]
        }
'''

# Step 1: Remove "params." and "self."
equations = re.sub(r"params\.", "", equations)
equations = re.sub(r"self\.", "", equations)

# Step 2: Replace Python exponentiation (**) with Mathematica exponentiation (^)
equations = re.sub(r"\*\*", "^", equations)

# Step 3: Replace "sin" and "cos" with "Sin" and "Cos" and convert their arguments to parentheses
equations = re.sub(r"(sin|cos)\(([^()]*)\)", lambda m: m.group(1).capitalize() + "[" + m.group(2) + "]", equations)
equations = re.sub(r'(\w+)\s*\[\s*(\d+)\s*\]', r'\1\2', equations)

print(equations)

print("\nPython code:")

math_expr =  '''
(l1*(a0*l1*m1 + a0*l1*m2 + a0*l1*m3 + a1*l2*(m2 + m3)*Cos[x0 - x1] + 
a2*l2*m3*Cos[x0 - x2] + g*m1*Sin[x0] + g*m2*Sin[x0] + g*m3*Sin[x0] + 
l2*m2*x4^2*Sin[x0 - x1] + l2*m3*x4^2*Sin[x0 - x1] + l2*m3*x5^2*Sin[x0 
- x2]), l2*(a1*l2*m2 + a1*l2*m3 + a0*l1*(m2 + m3)*Cos[x0 - x1] + 
a2*l2*m3*Cos[x1 - x2] - l1*m2*x3^2*Sin[x0 - x1] - l1*m3*x3^2*Sin[x0 - 
x1] + g*m2*Sin[x1] + g*m3*Sin[x1] + l2*m3*x5^2*Sin[x1 - x2]), 
l2*m3*(a2*l2 + a0*l1*Cos[x0 - x2] + a1*l2*Cos[x1 - x2] - 
l1*x3^2*Sin[x0 - x2] - l2*x4^2*Sin[x1 - x2] + g*Sin[x2]))
'''

# Step 1: Replace "^" with Python "**" for exponentiation
math_expr = math_expr.replace("^", "**")

# Step 2: Introduce `self.` and `params.` where appropriate
# Define patterns for the variables that should be prefixed with "self." or "params."

# Function to add prefixes to specific variables
def add_prefixes(expr):
    # Add `self.` prefix to `x` and `u` (assuming these belong to self)
    expr = re.sub(r'([ax])(\d+)', r'\1[\2]', expr)
    expr = re.sub(r'\bx\[(\d+)\]', r'self.x[\1]', expr)  # e.g. x[0] -> self.x[0]
    expr = re.sub(r'\ba\[(\d+)\]', r'self.a[\1]', expr)  # e.g. a[0] -> self.a[0]
    
    # Add `params.` prefix to the remaining variables (assuming these belong to params)
    param_list = ["g", "l1", "l2", "l3", "m1", "m2", "m3"]  # List of known parameters
    for param in param_list:
        expr = re.sub(rf'\b{param}\b', f'params.{param}', expr)
    
    return expr

# Apply the function to the expression
math_expr = add_prefixes(math_expr)
# math_expr = re.sub(r"(Sin|Cos)\[([^()]*)\]", lambda m: m.group(1).lower() + "(" + m.group(2) + ")", math_expr)

# Output the final Python expression with correct syntax
print(math_expr)