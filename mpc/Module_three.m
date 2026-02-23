%% ================================================================
% Module 3: Robust Tube-based MPC (visualization)
% Demonstrates nominal vs actual trajectory, tube boundary, 
%  and effect of disturbance and feedback gain K.
% Students may change Qerror, Rerror, disturbance_bound
% =================================================================

clear; 
clc; 
close all;

%% System matrices (2D state vector)

A = [1  0.1; 0   1];   % dynamic matrix
B = [0; 1];            % input matrix

nx = size(A,1);        % size of state vector
nu = size(B,2);        % size of control input vector 

%% Tube MPC parameters

Np = 10;                        % MPC prediction horizon
Q = diag([5,1]);
R = 0.1;

% --------------------------------------
% --------------------------------------
% Ancillary LQR for the error dynamics (u = -K e) to ensure stabilizable K

Qerror = diag([10, 1]);             % tuneable: penalize position >> velocity (students may vary)
                                    % try for instance [10,1], [20,1], [5,1] to see tube changes

Rerror = 0.9;                       % tuneable: input penalty  (students may vary)
                                    % try for instance 0.1, 0.5, 1.0

[Kstable, ~ , eigCL] = dlqr(A, B, Qerror, Rerror);
disp('eig(A-BK) = '), disp(eigCL.')
% --------------------------------------
% --------------------------------------

K = Kstable;                        % ancillary feedback gain 
disturbance_bound = 0.5;            % students vary to: 0.1, 0.3, 0.5, ...

Nsim = 40;                          % number of simulation steps
x_nom = zeros(nx , Nsim+1);         % stores predicted nominal states
x_act = zeros(nx , Nsim+1);         % stores predicted actual states

x0 = [-3; 2];                       % initial state
x_nom(:,1) = x0;                    % initializing nominal state with x0
x_act(:,1) = x0;                    % initializing actual state with x0

%% Build prediction matrices

Phi = zeros(nx*Np, nx);
Gamma = zeros(nx*Np, nu*Np);

for i = 1 : Np
    Phi((i-1)*nx+1:i*nx, :) = A^i;
    for j = 1:i
        Gamma((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = A^(i-j)*B;
    end
end

Qbar = kron(eye(Np), Q);
Rbar = kron(eye(Np), R);

%% Nominal Tube MPC loop
for k = 1 : Nsim

    xk_nom = x_nom(:,k);

    % MPC QP for nominal system
    H = Gamma'*Qbar*Gamma + Rbar;
    f = (Gamma'*Qbar*Phi*xk_nom)';

    uopt = quadprog(H, f, [], [], [], [], [], [], [], ...
        optimoptions('quadprog','Display','off'));

    if isempty(uopt)
        u_nom = 0;
        warning('Nominal QP infeasible.')
    else
        u_nom = uopt(1);
    end

    % Actual system input with disturbance feedback
    e = x_act(:,k) - x_nom(:,k);
    u_act = u_nom - K*e;

    % Apply disturbance to actual system
    w = disturbance_bound*(2*rand(2,1)-1);          % bounded disturbance
    x_nom(:,k+1) = A*x_nom(:,k) + B*u_nom;          % evolution of nominal state
    x_act(:,k+1) = A*x_act(:,k) + B*u_act + w;      % evolution of actual state
end

%% Plot

figure; 
hold on; 
grid on;
box on; 
plot(x_nom(1,:), x_nom(2,:), 'b-', 'LineWidth', 2)
plot(x_act(1,:), x_act(2,:), 'r-', 'LineWidth', 2)

% Plot tube around nominal trajectory
for k = 1 : 5 : Nsim                  % If your plot is too cluttered or not many cross sections for the tube show, 
                                      %   you can change 5 to a different value 
                                      
    viscircles(x_nom(:,k)', disturbance_bound*norm(K), 'Color',[0 0.6 0], 'LineWidth',0.2);
end

legend('Nominal trajectory','Actual trajectory','Tube boundary')
xlabel('$x_1$','Interpreter','latex'); 
ylabel('$x_2$','Interpreter','latex')
title('Module 3: Tube MPC visualization')