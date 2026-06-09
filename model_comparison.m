close all; clear; clc;


%% --------------------------- GLOBAL PARAMETERS ---------------------------
P = struct();

% Domain
P.Lx = 1.0;            % domain width  
P.Ly = 1.0;            % domain height

% Time discretisation
P.T  = 2.5;            % final time
P.dt = 1e-3;           % time step
P.Nt = round(P.T/P.dt);% no timesteps

% Cell population
P.Ncells = 250;             % no of MCCs
P.init_shape = 'disc';     % 'disc' or 'strip': shape that the cells form in IC
P.init_disc_radius = 0.10;  % if 'disc' is selected
P.init_strip_xmax  = 0.15;  % if 'strip is selected

% Haptotaxis coeffs
P.mu = 0.2; %0.25;          % drift strength coeff
P.ecm_grad = [1; 0];        % (constant) ECM gradient direction

% Baseline CPP (compound poisson)
P.lambda0 = 6.0;            % baseline Poisson rate (events per unit time)
P.jump_sigma = 0.012;       % baseline jump length scale
P.jump_dist = 'gauss';      % 'gauss' or 'exp' (length with random direction)

% Adhesion geometry
P.Rint = 0.03;              % interaction radius
P.cell_wh_ratio = 0.6;      % width/height relative to length
P.S_surface = 3.12;         % total cuboid surface area in the normalised units
P.da_mech = P.Rint;         % mechanical adhesion radius
P.dr_mech = 0.85*P.Rint;    % mechanical repulsion radius

% Adhesion / bonds
P.kappa = 45;               % dimensionless scaling constant for jump reduction
P.ncatch = 400;             % number of catch bonds 
P.nslip = 400;              % number of slip bonds
P.Fswitch = 10.18;          % pN, catch->slip "all transitioned" threshold in the draft
P.A_mech  = 0.02;           % repulsion/adhesion strength mechanical model

% Bell-model parameters for N-cadherin slip bonds
P.kBT_pN_nm = 4.28;         % k_B T in pN*nm, approximately at 37 C
P.koff0_s   = 1.0;          % s^{-1}, effective unstressed off-rate for N-cadherin
P.xbeta_nm  = 0.77;         % nm, reactive compliance for N-cadherin
P.Fmax_bell = 40.0;         % pN, numerical upper cap for Bell-law evaluation


% For (S2) controlled-force tests
P.F_test = [0, 3, 6, 9, 11, 14];  % pN (spans below/near/above transition)

% Number of experiment repetitions
P.nRep = 4; % 20;                   

% Plotting
P.save_plots = false;


%% --------------------------- SIMULATION (S1) ----------------------------
snapTimes = linspace(0,P.T,6);
nF = numel(P.F_test);

% preallocate separate cell arrays
results_baseline = cell(P.nRep,1);
results_adhesion  = cell(P.nRep,1);
results_mechanical  = cell(P.nRep,1);

tic
X0 = initialise_cells(P);
parfor r = 1:P.nRep
    results_baseline{r} = simulate_MCC(P, X0, 'baseline', snapTimes, [], true);
    results_adhesion{r} = simulate_MCC(P, X0, 'adhesion', snapTimes, [], true);
    results_mechanical{r} = simulate_MCC(P, X0, 'mechanical', snapTimes, [], true);
end

% assemble struct after parfor
results_S1.baseline = results_baseline;
results_S1.adhesion  = results_adhesion;
results_S1.mechanical  = results_mechanical;


toc

clear Pconst

% Example snapshots 
figure('Color','w'); hold on
colB = [0.0000 0.4470 0.7410];   % baseline: blue
colA = [0.8500 0.3250 0.0980];   % adhesion: orange
colC = [0.9290 0.6940 0.1250];   % mechanical: yellow

for repShow = 1:P.nRep
    outB = results_S1.baseline{repShow};
    outA = results_S1.adhesion{repShow};
    outC = results_S1.mechanical{repShow};

    plot(outB.tvec, outB.R, 'LineWidth',1.5, 'Color', colB);
    plot(outA.tvec, outA.R, 'LineWidth',1.5, 'Color', colA);
    plot(outC.tvec, outC.R, 'LineWidth',1.5, 'Color', colC);
end
hold off;
xlabel('Time'); ylabel('R(t) (radius of gyration)');
legend('baseline','adhesion','mechanical','Location','best');
grid on; %title('Population radius vs time');
drawnow;

% choose one repretition to plot details
repShow = 1;
plot_snapshots_new(results_S1.baseline{repShow}, results_S1.adhesion{repShow}, results_S1.mechanical{repShow}, P, snapTimes);


% Plot MSD over time for all nRep
meanMSD_B = zeros(P.Nt+1,1);
meanMSD_A = zeros(P.Nt+1,1);
meanMSD_C = zeros(P.Nt+1,1);
for r = 1:P.nRep
    outB = results_S1.baseline{r};
    outA = results_S1.adhesion{r};
    outC = results_S1.mechanical{r};

    meanMSD_B = meanMSD_B + outB.MSD;
    meanMSD_A = meanMSD_A + outA.MSD;
    meanMSD_C = meanMSD_C + outC.MSD;
end



%% --------------------------- SIMULATION (new S2) ----------------------------
% Controlled-force MSD comparison at final time

P2 = P;
P2.Ncells = 80;
P2.T  = 2.5;
P2.dt = 1e-3;
P2.Nt = round(P2.T/P2.dt);

% Remove drift to isolate motility statistics
P2.mu = 0.0;
P2.ecm_grad = [0;0];

nF = numel(P2.F_test);
nRep = 20;   % use 20 for the manuscript figure if computationally feasible

MSD_baseline   = zeros(nF, nRep);
MSD_adhesion   = zeros(nF, nRep);
MSD_mechanical = zeros(nF, nRep);

P2const = parallel.pool.Constant(P2);

tic;

for iF = 1:nF
    F = P2.F_test(iF);

    parfor r = 1:nRep

        P_local = P2const.Value;

        % Use the same initial condition for the three models
        rng(1000 + r, 'combRecursive');
        X0 = initialise_cells(P_local);

        % Baseline: independent of F
        rng(2000 + r, 'combRecursive');
        outB = simulate_MCC(P_local, X0, 'baseline', [], F, true);

        % Adhesion: depends on F through F_override
        rng(3000 + r, 'combRecursive');
        outA = simulate_MCC(P_local, X0, 'adhesion', [], F, true);

        % Mechanical: independent of F, included as force-based reference
        rng(4000 + r, 'combRecursive');
        outC = simulate_MCC(P_local, X0, 'mechanical', [], F, true);

        MSD_baseline(iF,r)   = mean(sum((outB.X - X0).^2, 2));
        MSD_adhesion(iF,r)   = mean(sum((outA.X - X0).^2, 2));
        MSD_mechanical(iF,r) = mean(sum((outC.X - X0).^2, 2));
    end
end

toc;
clear P2const

%% Plot updated Figure 13

Lscale = 1;

MSD_baseline_plot   = (Lscale^2) * MSD_baseline;
MSD_adhesion_plot   = (Lscale^2) * MSD_adhesion;
MSD_mechanical_plot = (Lscale^2) * MSD_mechanical;

figure('Color','w'); hold on

errorbar(P2.F_test, mean(MSD_baseline_plot,2), ...
    std(MSD_baseline_plot,[],2), '-o', 'LineWidth', 1.5);

errorbar(P2.F_test, mean(MSD_adhesion_plot,2), ...
    std(MSD_adhesion_plot,[],2), '-o', 'LineWidth', 1.5);

errorbar(P2.F_test, mean(MSD_mechanical_plot,2), ...
    std(MSD_mechanical_plot,[],2), '-o', 'LineWidth', 1.5);

xlabel('Force F (pN)');

if Lscale == 1
    ylabel('MSD at T');
else
    ylabel('MSD at T (\mum^2)');
end

legend('baseline','adhesion','mechanical','Location','best');
grid on; box on;
drawnow;

%% =============================== additional FUNCTIONS ===============================

%% ------------------------------------------
function out = simulate_MCC(P, X0, mode, snapTimes, F_override, recordR)

X = X0;
N = size(X,1);

% Snapshot storage
snapIdx = [];
if ~isempty(snapTimes)
    snapIdx = unique(max(1, min(P.Nt, round(snapTimes/P.dt))));
end
snapshots = cell(numel(snapIdx),1);

% Radius storage (initialise AFTER X is defined)
if recordR
    R    = zeros(P.Nt+1,1);
    tvec = (0:P.Nt)' * P.dt;

    xbar = mean(X,1);
    R(1) = sqrt(mean(sum((X - xbar).^2,2)));

    MSD = zeros(P.Nt+1,1);
    MSD(1) = mean(sum((X - X0).^2, 2));
else
    R = [];
    tvec = [];
end
% time loop (per timestep)
for t = 1:P.Nt
    % ECM and drift
    gradv = ecm_gradient(P, X);
    X = X + (P.mu * gradv) * P.dt;

    % Cell-Cell interactions on the DRIFT
    switch lower(mode)
        case 'mechanical'
            % da = P.Rint;
            % dr = P.Rint*0.7;
            
            Kall = zeros(N,2); % to store all 

            % slope = 1.25;
            % F_crit = 32*slope*(da-dr)*(3*da^2 + 4*da*dr + 3*dr^2)/(5*(11*slope+6)*dr^3);
            % Fr_c = 5e-3;
            % Fa_c = 0.9*Fr_c/F_crit;

            % Fa_c = 1e-3;
            
            
            driftMag1 = vecnorm(P.mu*gradv,2,2);
            F1 = rescale(driftMag1, 0, P.Fswitch*1.3);
            % F_scale = adhesion_jump_scale(P, X, F1);

            for i = 1:N
                K = zeros(1,2);
                j = 1;
                % Fa_c = P.A_mech;   % F_scale(i);
                % [Fa_c, F_scale(i)]

                while j<=N 
                    Xij = X(i,:)-X(j,:); dX = norm(Xij);
                    if j==i || dX>=P.da_mech || dX<=1e-12
                        j = j+1;
                        continue
                    end
                     
                    % K = K + (dX>0 & dX<dr/2).*(-Fr_c*(dr./(2*dX)).^(3-2*slope) .* Xij./dX) + ...
                    %         (dX>=dr/2 & dX<dr).*(2*Fr_c*(dX-dr)/dr .* Xij./dX) + ... 
                    %         (dX>=dr & dX<=da).*(-4*Fa_c*(dX-da).*(dX-dr)/((da-dr)^2) .* Xij./dX);
        
                    % K = K + (dX>=dr & dX<=da).*(-4*Fa_c*(dX-da).*(dX-dr)/((da-dr)^2) .* Xij./dX);

                    K = K + 4*P.A_mech*(dX-P.da_mech)*(dX-P.dr_mech)/((P.da_mech-P.dr_mech)^2)*Xij./dX;
                    
                    j = j+1;
                end

                % add the total force on every cells
                Kall(i,:) = K;

            end

            X = X + Kall * P.dt;

        case 'adhesion'
            % X = X;

        case 'baseline'
            % X = X;

    end
    

    % CPP jumps
    switch lower(mode)
        case 'baseline'
            lambda = P.lambda0 * ones(N,1);
            jumpScale = ones(N,1);

        case 'adhesion'
            if ~isempty(F_override)
                F = F_override * ones(N,1);
            else
                driftMag = vecnorm(P.mu*gradv,2,2);
                F = rescale(driftMag, 0, P.Fswitch*1.3);
            end

            gammaSum = obstruction_sum(P, X);
            lambda = P.lambda0 .* max(0, 1 - gammaSum);

            jumpScale = adhesion_jump_scale(P, X, F);
         
        case 'mechanical'
            lambda = P.lambda0 * ones(N,1);
            jumpScale = ones(N,1);

        otherwise
            error('Unknown mode.');
    end

    nEvents = poissrnd(lambda * P.dt);

    for i = 1:N
        if nEvents(i) == 0, continue; end
        dJ = [0,0];
        for k = 1:nEvents(i)
            dJ = dJ + sample_jump(P);
        end
        X(i,:) = X(i,:) + jumpScale(i) * dJ;
    end

    % BCs
    X = apply_boundary(P, X);

    % Record radius
    if recordR
        xbar = mean(X,1);
        R(t+1) = sqrt(mean(sum((X - xbar).^2,2)));
        MSD(t+1) = mean(sum((X - X0).^2, 2));
    end

    % Save snapshots
    if ~isempty(snapIdx)
        idx = find(snapIdx == t, 1);
        if ~isempty(idx)
            snapshots{idx} = X;
        end
    end
end

out = struct();
out.X = X;
out.snapIdx = snapIdx;
out.snapshots = snapshots;
out.tvec = tvec;
out.R = R;
out.MSD = MSD;

end

% ---------- 
function gradECM= ecm_gradient(P, X)
% ECM gradient field. For clean interpretation we keep it simple.
    N = size(X,1);
    g = P.ecm_grad(:);
    % gradECM = repmat(g(:).', N, 1);
    % gradECM = (X + 1e-1*ones(N, 2, 1)) .* repmat(g(:).', N, 1);
    gradECM = 0.5*X.* repmat(g(:).', N, 1) + repmat(g(:).', N, 1);
end

%----------------------------------
function X0 = initialise_cells(P)
% Initialise MCC positions.
% Returns X0 as an Ncells-by-2 array.

N = P.Ncells;

switch lower(P.init_shape)
    case 'disc'
        % Disc near the left boundary
        centre = [0.12, 0.5];
        % centre = [0.5, 0.5];
        R = P.init_disc_radius;
        theta = 2*pi*rand(N,1);
        rad = R*sqrt(rand(N,1));
        X0 = [centre(1) + rad.*cos(theta), centre(2) + rad.*sin(theta)];

    case 'strip'
        % Vertical strip close to left boundary
        X0 = [P.init_strip_xmax*rand(N,1), P.Ly*rand(N,1)];

    otherwise
        error('Unknown init_shape.');
end

% Keep within bounds
X0(:,1) = min(max(X0(:,1),0),P.Lx);
X0(:,2) = min(max(X0(:,2),0),P.Ly);
end


%% ----------------------------------
function gammaSum = obstruction_sum(P, X)
% Sum of obstruction fractions gamma_{k,i} over all neighbours within interaction radius.


N = size(X,1);
gammaSum = zeros(N,1);

for k = 1:N
    dx = X(:,1) - X(k,1);
    dy = X(:,2) - X(k,2);
    d  = hypot(dx,dy);

    nbr = find(d > 0 & d < P.Rint);
    if isempty(nbr), continue; end

    
    % A_{k,i} \in [0, Amax], with Amax chosen as the "long face" area (0.6) times a factor.
    Amax = 0.6; % long face area in the normalised cuboid model
    Aki  = Amax * max(0, 1 - d(nbr)/P.Rint);

    gammaSum(k) = sum(Aki) / P.S_surface;
end

% Prevent rates going negative in dense packing
gammaSum = min(gammaSum, 1.0);
end

%% ----------------------------------
function s = adhesion_jump_scale(P, X, F)

N = size(X,1);
s = ones(N,1);

% Determine neighbours (potential adhesion partners)
for k = 1:N
    dx = X(:,1) - X(k,1);
    dy = X(:,2) - X(k,2);
    d  = hypot(dx,dy);
    nbr = find(d > 0 & d < P.Rint);

    if isempty(nbr)
        s(k) = 1.0;
        continue;
    end

    alphaSumInv = 0;

    for j = nbr(:).'
        % Catch-slip switch time: we use a simple instantaneous mapping:
        % - If F below switch, mixture possible; above switch, mostly slip.
        % You can replace this with the px,ss(F,t) logic if you explicitly track contact time.

        Fj = F(k); % force used for bonds of cell k (per-cell proxy)

        % Sample mean bond lifetimes (per bond-type) using Gamma with prescribed mean/SD
        [Lc, sigc] = catch_stats(P, Fj);
        [Ls, sigs] = slip_stats_bell(P, Fj);

        % Approximate mean of many bonds: Xbar ~ Normal(L, sig^2/n)
        Xbar_c = max(1e-6, Lc + sqrt(sigc^2/max(P.ncatch,1))*randn());
        Xbar_s = max(1e-6, Ls + sqrt(sigs^2/max(P.nslip,1))*randn());

        alpha = P.ncatch * Xbar_c + P.nslip * Xbar_s; % total lifetime proxy

        alphaSumInv = alphaSumInv + (P.kappa / alpha);
    end

    % We cap to avoid numerical extremes.
    s(k) = min(1.0, max(0.02, alphaSumInv));
end
end

%% ----------------------------------
function [Lc, sigc] = catch_stats(P, F)
% Mean and SD for catch bonds as functions of force (from the draft forms).
% Valid (as written) for F <= Fswitch. We extend smoothly beyond by holding at switch.

F0 = min(F, P.Fswitch);

% From the manuscript (rounded values)
Lc   = 2.36e-6 * exp(F0) + 0.0257;
sigc = 1.33e-6 * exp(F0) + 0.0260;
end


%% ----------------------------------
function [Ls, sigs] = slip_stats_bell(P, F)
% Mean and SD for slip-bond lifetimes using Bell's model.
% F is the effective scalar force per bond in pN.
F = max(F, 0);
F = min(F, P.Fmax_bell);
koff = P.koff0_s .* exp((F .* P.xbeta_nm) ./ P.kBT_pN_nm);
Ls   = 1 ./ koff;
sigs = 1 ./ koff;
% Numerical guard
Ls   = max(Ls, 1e-6);
sigs = max(sigs, 1e-6);
end

%% ----------------------------------
function dJ = sample_jump(P)
% Sample one 2D jump for the CPP.

switch lower(P.jump_dist)
    case 'gauss'
        dJ = P.jump_sigma * randn(1,2);

    case 'exp'
        theta = 2*pi*rand();
        len = exprnd(P.jump_sigma);
        dJ = len * [cos(theta), sin(theta)];

    otherwise
        error('Unknown jump_dist.');
end
end

%% ----------------------------------
function X = apply_boundary(P, X)
% Apply (reflect) boundary conditions.
    X(:,1) = reflect_coord(X(:,1), 0, P.Lx);
    X(:,2) = reflect_coord(X(:,2), 0, P.Ly);

    %----------------------------- (nested)
    function x = reflect_coord(x, a, b)
    % Reflect coordinates into [a,b] by mirror reflection.
    range = b - a;
    x = x - a;

    % Map to [0, 2*range) then reflect
    x = mod(x, 2*range);
    mask = x > range;
    x(mask) = 2*range - x(mask);

    x = x + a;
    end
end


%% -----------------------------
function plot_snapshots_new(outB, outA, outC, P, snapTimes)
% Visual comparison of snapshots: baseline vs adhesion.

if isempty(outB.snapshots) || isempty(outA.snapshots) || isempty(outC.snapshots), return; end

figure('Color','w');
nS = numel(outB.snapshots);
for i = 1:nS
    XB = outB.snapshots{i};
    XA = outA.snapshots{i};
    XC = outC.snapshots{i};
    
    % USE THIS FOR PLOTTING THE GRADIENT EXPERIMENT
    x_grad = 0:0.01:P.Lx;
    y_grad = 0:0.01:P.Ly;
    [xC, yC] = meshgrid(x_grad,y_grad);
    % LL = size(xC);
    % gradECM = xC + 1e-1*ones(LL);
    % gradECM = ones(LL);

    Xgrid = [xC(:), yC(:)];
    gradv = ecm_gradient(P, Xgrid);
    gradECM = reshape(gradv(:,1), size(xC));

    
    subplot(3,nS,i);
    imagesc(x_grad, y_grad, gradECM); flipud(colormap(pink)); hold on;
    plot(XB(:,1), XB(:,2), '.b', 'MarkerSize',2); hold on;
    [hullB,areaB] = convhull(XB);
    plot(XB(hullB,1), XB(hullB,2), 'k--', 'LineWidth', 0.5)
    % legend(sprintf('Area=%.5f', areaB))
    colorbar;
    axis([0 P.Lx 0 P.Ly]); axis square;
    % title(sprintf('Baseline t=%.2f', snapTimes(i)));
    set(gca,'XTick',[],'YTick',[]);

    subplot(3,nS,nS+i);
    imagesc(x_grad, y_grad, gradECM); flipud(colormap(pink)); hold on;
    plot(XA(:,1), XA(:,2), '.b', 'MarkerSize',2); hold on;
    [hullA,areaA] = convhull(XA);
    plot(XA(hullA,1), XA(hullA,2), 'k--', 'LineWidth', 0.5)
    % legend(sprintf('Area=%.5f', areaA))
    colorbar;
    axis([0 P.Lx 0 P.Ly]); axis square;
    % title(sprintf('Adhesion t=%.2f', snapTimes(i)));
    set(gca,'XTick',[],'YTick',[]);

    subplot(3,nS,2*nS+i);
    imagesc(x_grad, y_grad, gradECM); flipud(colormap(pink)); hold on;
    plot(XC(:,1), XC(:,2), '.b', 'MarkerSize',2); hold on;
    [hullC,areaC] = convhull(XC);
    plot(XC(hullC,1), XC(hullC,2), 'k--', 'LineWidth', 0.5)
    % legend(sprintf('Area=%.5f', areaC))
    colorbar;
    axis([0 P.Lx 0 P.Ly]); axis square;
    % title(sprintf('Mechanical t=%.2f', snapTimes(i)));
    set(gca,'XTick',[],'YTick',[]);
end
end
