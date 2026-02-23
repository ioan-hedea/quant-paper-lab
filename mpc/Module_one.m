%% ================================================================
% Module 1: Prediction and system behavior
% Linear system: x(k+1) = A x(k) + B u(k)
% Students change x0, disturbance magnitude & timing, and u.
% =================================================================

clear; 
clc; 
close all;

%% System matrices (2D state vector)

A = [1  0.1; 0   1];   % dynamic matrix
B = [0; 1];            % input matrix

%% Simulation settings

Nsim = 40;             % number of simulation steps
x0  = [-2; 3];         % initial state (students may change)
u   = 0.1;             % constant control input (students may change)

% Add disturbance at a chosen time step:

disturbance_step = 15;  % time step when disturbance is exerted (students may change)
d = [1; -1];            % disturbance vector (students may change)

%% Simulate open-loop trajectory

x = zeros(2 , Nsim + 1);    % initialize predicted state vector with zeros
x(:,1) = x0;                % insert the actual initial state in the first column

for k = 1 : Nsim
    if k == disturbance_step
        x(:,k) = x(:,k) + d;
    end
    x(:,k+1) = A*x(:,k) + B*u;
end

%% Plot

figure; 
hold on; 
grid on;
plot(0 : Nsim, x(1 , :), 'LineWidth', 2)
plot(0 : Nsim, x(2,:), 'LineWidth', 2)
xlabel('$k$','Interpreter','latex'); 
ylabel('State components')
legend('$x_1$','$x_2$','Interpreter','latex')
title('Module 1: Open-loop state prediction')


%% -------- Closed-loop simulation with simple state-feedback K --------
% Activate the following part of the code only for Question 5 (closed-loop preview)

% % Compute feedback gain K via discrete LQR (requires Control System Toolbox)
% % LQR weights for error/state damping (tune if you prefer)
% Q_lqr = diag([1.0, 0.2]);   % penalize position more than velocity
% R_lqr = 0.5;                % penalize input effort moderately
% % Compute stabilizing K (for discrete-time A,B)
% K = dlqr(A, B, Q_lqr, R_lqr);           % K is of size 1 x 2
% 
% % Closed-loop reference (optional): regulate to the origin by default.
% x_ref = [0; 0];   
% 
% x_cl = zeros(2, Nsim+1);
% x_cl(:,1) = x0;
% 
% for k = 1:Nsim
%     % Apply the same disturbance timing/magnitude to compare fairly
%     if k == disturbance_step
%         x_cl(:,k) = x_cl(:,k) + d;
%     end
% 
%     % State-feedback control (regulation to x_ref)
%     u_fb = -K * (x_cl(:,k) - x_ref);   % stabilizing feedback
% 
%     % Closed-loop update
%     x_cl(:,k+1) = A * x_cl(:,k) + B * u_fb;
% end
% 
% % -------------------- Plot: open-loop versus closed loop ---------------------
% kvec = 0:Nsim;
% 
% figure('Color','w'); hold on; grid on;
% 
% % Open-loop
% plot(kvec, x(1,:), 'b-',  'LineWidth', 2, 'DisplayName', '$x_1$ (open-loop)');
% plot(kvec, x(2,:), 'r-',  'LineWidth', 2, 'DisplayName', '$x_2$ (open-loop)');
% 
% % Closed-loop
% plot(kvec, x_cl(1,:), 'b--', 'LineWidth', 2, 'DisplayName', '$x_1$ (closed-loop)');
% plot(kvec, x_cl(2,:), 'r--', 'LineWidth', 2, 'DisplayName', '$x_2$ (closed-loop)');
% 
% % Disturbance marker
% yl = ylim;
% plot([disturbance_step disturbance_step], yl, 'k:', 'LineWidth', 1.5, ...
%      'DisplayName', 'disturbance time');
% 
% xlabel('$k$','Interpreter','latex'); 
% ylabel('State components','Interpreter','latex');
% legend('Interpreter','latex','Location','best');
% title('Module 1: Open-loop vs. closed-loop with disturbance','Interpreter','latex');
