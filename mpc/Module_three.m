%% ================================================================
% Module 3: Robust Tube-based MPC (constrained implementation)
% This script solves a constrained nominal MPC and applies
% ancillary feedback u = v - K(x-z) to reject bounded disturbances.
% Students may change Qerror, Rerror, disturbance_bound
% =================================================================

clear;
clc;
close all;

%% -------------------- System matrices (2D state vector) ----------

A = [1  0.1; 0   1];   % dynamic matrix
B = [0; 1];            % input matrix

nx = size(A,1);        % size of state vector
nu = size(B,2);        % size of control input vector

%% -------------------- MPC weights and horizon ---------------------

Np = 10;                        % MPC prediction horizon
Q = diag([5,1]);
R = 0.1;
Nsim = 40;                      % number of simulation steps

% Optional terminal weight
try
    P = dare(A, B, Q, R);
catch
    P = Q;
end

%% -------------------- Constraints ---------------------------------
xmin = [-5; -5];
xmax = [ 5;  5];
umin = -0.8;
umax =  0.8;

%% -------------------- Ancillary error feedback K ------------------
% Error dynamics: e+ = (A-BK)e + w

Qerror = diag([10, 1]);         % students may vary
Rerror = 0.9;                   % students may vary
[K,~,eigCL] = dlqr(A, B, Qerror, Rerror);
Acl = A - B*K;
disp('eig(A-BK) = ');
disp(eigCL.');

%% -------------------- Disturbance and tube tightening -------------
disturbance_bound = 0.2;        % assignment base run default; try 0.1, 0.3, 0.5
w_inf = disturbance_bound * ones(nx,1);  % box disturbance: |w_i| <= disturbance_bound

% Compute component-wise RPI bound r_inf for |e| <= r_inf:
% r_{k+1} = |Acl| r_k + w_inf
r_inf = zeros(nx,1);
for it = 1:500
    r_next = abs(Acl) * r_inf + w_inf;
    if norm(r_next - r_inf, inf) < 1e-10
        r_inf = r_next;
        break;
    end
    r_inf = r_next;
end

% Input tightening term: |K e| <= kappa for |e| <= r_inf
kappa = abs(K) * r_inf;   % scalar because nu = 1

xmin_tight = xmin + r_inf;
xmax_tight = xmax - r_inf;
umin_tight = umin + kappa;
umax_tight = umax - kappa;

if any(xmin_tight >= xmax_tight) || any(umin_tight >= umax_tight)
    error(['Tube tightening infeasible. Reduce disturbance_bound or adjust constraints. ' ...
           'Current tightened ranges are empty.']);
end

%% -------------------- Build prediction matrices -------------------

Phi = zeros(nx*Np, nx);
Gamma = zeros(nx*Np, nu*Np);

for i = 1 : Np
    Phi((i-1)*nx+1:i*nx, :) = A^i;
    for j = 1:i
        Gamma((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = A^(i-j)*B;
    end
end

Qbar = kron(eye(Np), Q);
Qbar(end-nx+1:end, end-nx+1:end) = P;   % terminal block
Rbar = kron(eye(Np), R);

%% -------------------- Storage -------------------------------------
x0 = [-3; 2];
x_nom = zeros(nx, Nsim+1);   % nominal z
x_act = zeros(nx, Nsim+1);   % actual x
u_nom_hist = zeros(1, Nsim);
u_act_hist = zeros(1, Nsim);
qp_ok = false(1, Nsim);
e_norm_inf = zeros(1, Nsim+1);

x_nom(:,1) = x0;
x_act(:,1) = x0;
e_norm_inf(1) = norm(x_act(:,1)-x_nom(:,1), inf);

% Quadratic term is constant
H = Gamma' * Qbar * Gamma + Rbar;
H = (H + H')/2 + 1e-9*eye(size(H));
Aineq_u = [ eye(Np*nu); -eye(Np*nu) ];
bineq_u = [ umax_tight*ones(Np*nu,1); -umin_tight*ones(Np*nu,1) ];

%% -------------------- Tube MPC loop --------------------------------
for k = 1:Nsim

    xk_nom = x_nom(:,k);

    % Nominal predicted state constraints:
    % xmin_tight <= Phi*xk_nom + Gamma*V <= xmax_tight
    Aineq_x = [ Gamma; -Gamma ];
    bineq_x = [ kron(ones(Np,1), xmax_tight) - Phi*xk_nom;
               -kron(ones(Np,1), xmin_tight) + Phi*xk_nom ];

    % Combine state and input constraints
    Aineq = [Aineq_u; Aineq_x];
    bineq = [bineq_u; bineq_x];

    f = (Gamma' * Qbar * Phi * xk_nom)';

    Vopt = quadprog(H, f, Aineq, bineq, [], [], [], [], [], ...
        optimoptions('quadprog','Display','off'));

    if isempty(Vopt)
        % Safe fallback if QP becomes infeasible
        v0 = 0;
        warning('Nominal QP infeasible at step %d. Applying fallback v=0.', k);
    else
        qp_ok(k) = true;
        v0 = Vopt(1);
    end

    % Nominal input / dynamics
    u_nom = v0;
    x_nom(:,k+1) = A*x_nom(:,k) + B*u_nom;

    % Actual input with ancillary feedback
    e = x_act(:,k) - x_nom(:,k);
    u_act = u_nom - K*e;  % tube policy
    u_act = min(max(u_act, umin), umax); % enforce true actuator limits

    % Apply disturbance to actual system
    w = disturbance_bound * (2*rand(nx,1)-1);
    x_act(:,k+1) = A*x_act(:,k) + B*u_act + w;

    u_nom_hist(k) = u_nom;
    u_act_hist(k) = u_act;
    e_norm_inf(k+1) = norm(x_act(:,k+1)-x_nom(:,k+1), inf);
end

%% -------------------- Plots ----------------------------------------

figure('Color','w');
subplot(2,2,1); hold on;
grid on;
box on;
plot(x_nom(1,:), x_nom(2,:), 'b-', 'LineWidth', 2);
plot(x_act(1,:), x_act(2,:), 'r-', 'LineWidth', 2);

% Plot tube box cross-sections around selected nominal states
for k = 1:5:Nsim+1
    xk = x_nom(:,k);
    x1_left = xk(1) - r_inf(1);
    x2_low  = xk(2) - r_inf(2);
    rectangle('Position',[x1_left, x2_low, 2*r_inf(1), 2*r_inf(2)], ...
              'EdgeColor',[0 0.6 0], 'LineWidth',0.4);
end

legend('Nominal trajectory','Actual trajectory','Tube cross-sections','Location','best');
xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
title('Module 3: Tube MPC (state space)','Interpreter','latex');

subplot(2,2,2); hold on; grid on;
k = 0:Nsim-1;
stairs(k, u_nom_hist, 'b-', 'LineWidth', 1.8);
stairs(k, u_act_hist, 'r--', 'LineWidth', 1.8);
yline(umax,'k:','u_{max}');
yline(umin,'k:','u_{min}');
xlabel('$k$','Interpreter','latex');
ylabel('$u$','Interpreter','latex');
legend('u_{nom}','u_{act}','Location','best');
title('Inputs','Interpreter','latex');

subplot(2,2,3); hold on; grid on;
k2 = 0:Nsim;
plot(k2, e_norm_inf, 'm-', 'LineWidth', 1.8);
yline(max(r_inf), 'k--', 'max tube radius');
xlabel('$k$','Interpreter','latex');
ylabel('$||e_k||_{\infty}$','Interpreter','latex');
title('Error vs. tube size','Interpreter','latex');

subplot(2,2,4); hold on; grid on;
stem(0:Nsim-1, qp_ok, 'filled');
ylim([-0.1 1.1]);
yticks([0 1]); yticklabels({'infeasible','feasible'});
xlabel('$k$','Interpreter','latex');
ylabel('QP status','Interpreter','latex');
title('Nominal QP feasibility','Interpreter','latex');

fprintf('Tightened state bounds: x1 in [%.3f, %.3f], x2 in [%.3f, %.3f]\\n', ...
    xmin_tight(1), xmax_tight(1), xmin_tight(2), xmax_tight(2));
fprintf('Tightened input bounds: u in [%.3f, %.3f]\\n', umin_tight, umax_tight);
fprintf('QP feasible at %d/%d steps\\n', sum(qp_ok), Nsim);
