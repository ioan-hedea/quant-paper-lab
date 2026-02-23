%% ================================================================
% Module 2: MPC and horizon effect, with cost tuning, and constraints
% Students only need to change Np, Q, R 
% ================================================================

clear; 
clc; 
close all;

%% -------------------- System matrices (2D state) --------------------
A = [1  0.1;
      0  1];            % dynamics matrix
B = [0;
     1];                % input matrix

nx = size(A,1);         % state dimension
nu = size(B,2);         % input dimension

%% -------------------- Simulation settings ---------------------------
Nsim = 40;             % number of simulation steps
x0   = [-3; 2];        % initial state
Np   = 3;              % prediction horizon (students may change, e.g., to 3, 10, 30)

%% -------------------- Cost matrices (students tune) -----------------
Q = diag([5, 1]);      % increase Q -> stronger state regulation
R = 0.1;               % increase R -> smoother, more conservative inputs

%% -------------------- Constraints ----------------------------------
umin = -0.8;           % lower bound for u
umax =  0.8;           % upper bound for u
xmin = [-5; -5];       % lower bound for state (used for plotting safety)
xmax = [ 5;  5];       % upper bound for state (used for plotting safety)

%% -------------------- Storage --------------------------------------
x = zeros(nx, Nsim + 1);            % state trajectory (closed-loop)
u_applied = zeros(nu, Nsim);        % applied inputs
input_saturated = false(1, Nsim);   % constraint indicator: did input u hit its bounds at step k?
x(:,1) = x0;

%% -------------------- Build prediction matrices Phi, Gamma ----------

Phi   = zeros(nx*Np, nx);
Gamma = zeros(nx*Np, nu*Np);

for i = 1:Np
    Phi((i-1)*nx+1 : i*nx, :) = A^i;
    for j = 1:i
        Gamma((i-1)*nx+1 : i*nx, (j-1)*nu+1 : j*nu) = A^(i-j) * B;
    end
end

%% -------------------- MPC loop -------------------------------------
for k = 1:Nsim

    % Current state
    xk = x(:,k);

    % Quadratic cost for stacked input vector U:
    Qbar = kron(eye(Np), Q);
    Rbar = kron(eye(Np), R);

    H = Gamma' * Qbar * Gamma + Rbar;
    % Numerical regularization to ensure positive definiteness (helps quadprog)
    H = (H + H')/2 + 1e-8 * eye(size(H));

    f = (Gamma' * Qbar * Phi * xk)';  % linear term (constant term omitted)

    % Input constraints on all moves: umin <= U_i <= umax
    Aineq = [ eye(Np*nu); -eye(Np*nu) ];
    bineq = [ umax * ones(Np*nu,1); -umin * ones(Np*nu,1) ];

    % Solve QP: minimize 0.5*U'HU + f'U  subject to Aineq*U <= bineq
    Uopt = quadprog(H, f, Aineq, bineq, [], [], [], [], [], ...
                    optimoptions('quadprog','Display','off'));

    if isempty(Uopt)
        warning('QP infeasible at step %d. Applying u = 0.', k);
        u = 0;
        input_saturated(k) = false;
    else
        u = Uopt(1);  % receding-horizon: apply only the first move
        
        % Constraint indicator: did the applied input hit its bounds?
        input_saturated(k) = (abs(u - umax) < 1e-6) || (abs(u - umin) < 1e-6);
    end

    u_applied(k) = u;

    % Plant update
    x(:,k+1) = A * x(:,k) + B * u;

    % Optional: keep the plotted states within [xmin, xmax] for readability
    x(:,k+1) = min(max(x(:,k+1), xmin), xmax);

end

%% -------------------- Plots -----------------------------------------
k_state = 0:Nsim;
k_input = 0:Nsim-1;

figure('Color','w');

subplot(3,1,1);
plot(k_state, x(1,:), 'LineWidth', 1.8); hold on; grid on;
plot(k_state, x(2,:), 'LineWidth', 1.8);
ylabel('$x_1(k), x_2(k)$','Interpreter','latex');
legend('$x_1$','$x_2$','Interpreter','latex','Location','best');
title(sprintf('Module 2: MPC closed-loop (N_P = %d, Q = diag[%g,%g], R = %.3g)', ...
      Np, Q(1,1), Q(2,2), R), 'Interpreter','none');

subplot(3,1,2);
stairs(k_input, u_applied, 'LineWidth', 2); grid on;
ylabel('$u(k)$','Interpreter','latex');
yline(umax,'r--','u_{max}');
yline(umin,'r--','u_{min}');
title('Applied control input','Interpreter','latex');

subplot(3,1,3);
stem(k_input, input_saturated, 'filled'); grid on;
ylim([-0.1, 1.1]);
yticks([0 1]); yticklabels({'inactive','active'});
xlabel('$k$','Interpreter','latex'); ylabel('sat?','Interpreter','latex');
title('Input-constraint activation','Interpreter','latex');

%% -------- Including a terminal cost with matrix P obtained via DARE --------
% Activate the following part of the code only for Question 5 (effect of terminal cost)

% fprintf('\n[Add-on] Running a second experiment WITH terminal cost P...\n');
% 
% % --- Compute terminal weight P from DARE (LQR terminal weight) ---
% use_terminal_cost = true;
% try
%     [Pterm,~,~] = dare(A, B, Q, R);   % requires Control System Toolbox
% catch ME
%     warning('DARE failed or not available (%s). Falling back to P = Q.', ME.identifier);
%     use_terminal_cost = false;     % set false if you want to skip 
%     Pterm = Q;                     % simple fallback for demonstration
% end
% 
% % --- Rebuild Phi, Gamma for safety (in case Np changed) ---
% Phi_P   = zeros(nx*Np, nx);
% Gamma_P = zeros(nx*Np, nu*Np);
% for i = 1:Np
%     Phi_P((i-1)*nx+1:i*nx, :) = A^i;
%     for j = 1:i
%         Gamma_P((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = A^(i-j) * B;
%     end
% end
% 
% % --- Allocate storage for terminal-cost run ---
% x_P = zeros(nx, Nsim + 1);
% u_applied_P = zeros(nu, Nsim);
% input_saturated_P = false(1, Nsim);
% x_P(:,1) = x0;
% 
% % ====================== MPC with terminal cost ============================
% for k = 1:Nsim
%     xk = x_P(:,k);
% 
%     % Build stacked weights with terminal block P
%     Qbar_P = kron(eye(Np), Q);
%     if use_terminal_cost
%         Qbar_P(end-size(Q,1)+1:end, end-size(Q,1)+1:end) = Pterm;
%     end
%     Rbar_P = kron(eye(Np), R);
% 
%     % Cost matrices
%     H_P = Gamma_P' * Qbar_P * Gamma_P + Rbar_P;
%     H_P = (H_P + H_P')/2 + 1e-8*eye(size(H_P));  % regularization
% 
%     f_P = (Gamma_P' * Qbar_P * Phi_P * xk)';
% 
%     % Input constraints (same as baseline)
%     Aineq_P = [ eye(Np*nu); -eye(Np*nu) ];
%     bineq_P = [ umax*ones(Np*nu,1); -umin*ones(Np*nu,1) ];
% 
%     % Solve
%     Uopt_P = quadprog(H_P, f_P, Aineq_P, bineq_P, [], [], [], [], [], ...
%                       optimoptions('quadprog','Display','off'));
% 
%     if isempty(Uopt_P)
%         warning('[Add-on] QP infeasible at step %d (terminal cost run). Applying u = 0.', k);
%         uP = 0;
%         input_saturated_P(k) = false;
%     else
%         uP = Uopt_P(1);
%         input_saturated_P(k) = (abs(uP - umax) < 1e-6) || (abs(uP - umin) < 1e-6);
%     end
% 
%     u_applied_P(k) = uP;
%     x_P(:,k+1) = A*x_P(:,k) + B*uP;
% 
%     % Keep the plotted states within [xmin, xmax] for readability (same as baseline)
%     x_P(:,k+1) = min(max(x_P(:,k+1), xmin), xmax);
% end
% 
% % ===================== Comparison plots ======================
% k_state = 0:Nsim;
% k_input = 0:Nsim-1;
% 
% figure('Color','w');
% 
% subplot(3,1,1);
% plot(k_state, x(1,:), 'b-',  'LineWidth', 1.6); hold on; grid on;
% plot(k_state, x(2,:), 'r-',  'LineWidth', 1.6);
% plot(k_state, x_P(1,:), 'b--', 'LineWidth', 1.6);
% plot(k_state, x_P(2,:), 'r--', 'LineWidth', 1.6);
% ylabel('$x_1(k), x_2(k)$','Interpreter','latex');
% legend({'$x_1$ (no $P$)','$x_2$ (no $P$)','$x_1$ (with $P$)','$x_2$ (with $P$)'}, ...
%        'Interpreter','latex','Location','best');
% title(sprintf('Comparison: N_P=%d, Q=diag[%g,%g], R=%.3g (solid=no P, dashed=with P)', ...
%       Np, Q(1,1), Q(2,2), R), 'Interpreter','none');
% 
% subplot(3,1,2);
% stairs(k_input, u_applied,   'k-',  'LineWidth', 2); hold on; grid on;
% stairs(k_input, u_applied_P, 'm--', 'LineWidth', 2);
% ylabel('$u(k)$','Interpreter','latex');
% yline(umax,'r--','u_{max}');
% yline(umin,'r--','u_{min}');
% legend({'no $P$','with $P$'}, 'Interpreter','latex','Location','best');
% title('Applied control input (solid = no terminal cost, dashed = with terminal cost)','Interpreter','latex');
% 
% subplot(3,1,3);
% stem(k_input, input_saturated,   'filled','MarkerFaceColor',[0 0 0]); hold on; grid on;
% stem(k_input, input_saturated_P, 'filled','MarkerFaceColor',[0.85 0 0.85]);
% ylim([-0.1, 1.1]);
% yticks([0 1]); yticklabels({'inactive','active'});
% xlabel('$k$','Interpreter','latex'); ylabel('sat?','Interpreter','latex');
% legend({'no $P$','with $P$'}, 'Interpreter','latex','Location','best');
% title('Input-constraint activation (no $P$ vs with $P$)','Interpreter','latex');
% 
% fprintf('[Add-on] Done. Compare early inputs and saturation events (solid vs dashed).\n');