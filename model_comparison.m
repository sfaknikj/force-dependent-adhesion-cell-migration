%% FORCE-DEPENDENT ADHESION: NUMERICAL SIMULATIONS (BMB-ready package)
%   (S1) Baseline vs adhesion-enabled invasion (snapshots + invasion front)
%   (S2) Cell-scale motility statistics: MSD and effective diffusion vs force
%


clear; close all; clc;


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
P.Ncells = 250;        % no of MCCs
P.init_shape = 'disc'; % 'disc', 'strip': shape that the cells form in IC
P.init_disc_radius = 0.10;  % if 'disc' is selected
P.init_strip_xmax  = 0.15;  % if 'strip is selected

% Haptotaxis coeffs
P.mu = 0.25;           % drift strength coeff
P.ecm_grad = [1; 0];   % (constant) ECM gradient direction

% Baseline CPP (compound poisson)
P.lambda0 = 6.0;       % baseline Poisson rate (events per unit time)
P.jump_sigma = 0.012;  % baseline jump length scale
P.jump_dist = 'gauss'; % 'gauss' or 'exp' (length with random direction)

% Adhesion geometry
P.Rint = 0.03;         % interaction radius
P.cell_wh_ratio = 0.6; % width/height relative to length
P.S_surface = 3.12;    % total cuboid surface area in the normalised units

% Adhesion / bonds
P.kappa = 1.0;         % dimensionless scaling constant for jump reduction
P.ncatch = 400;        % number of catch bonds 
P.nslip = 400;         % number of slip bonds
P.Fswitch = 10.18;     % pN, catch->slip "all transitioned" threshold in the draft

% For (S2) controlled-force tests
P.F_test = [0, 3, 6, 9, 11, 14];  % pN (spans below/near/above transition)
P.nRep = 20;                      % replicates for statistics

% Plotting
P.save_plots = false;

%% --------------------------- SIMULATION (S1) ----------------------------
nRep_S1 = 10;
snapTimes = [0 0.2, 0.6, 1.2, 1.5, 2.0, 2.5];

% preallocate separate cell arrays (parfor-friendly)
results_baseline = cell(nRep_S1,1);
results_adhesion  = cell(nRep_S1,1);

Pconst = parallel.pool.Constant(P);

parfor r = 1:nRep_S1
    P_local = Pconst.Value;
    X0 = initialise_cells(P_local);

    results_baseline{r} = simulate_MCC(P_local, X0, 'baseline', snapTimes, [], true);
    results_adhesion{r} = simulate_MCC(P_local, X0, 'adhesion',  snapTimes, [], true);
end

% assemble struct after parfor
results_S1.baseline = results_baseline;
results_S1.adhesion  = results_adhesion;

clear Pconst

% Example snapshots (single replicate)
repShow = 1;
outB = results_S1.baseline{repShow};
outA = results_S1.adhesion{repShow};
plot_snapshots(results_S1.baseline{repShow}, results_S1.adhesion{repShow}, P, snapTimes);

figure('Color','w'); hold on
plot(outB.tvec, outB.R, 'LineWidth',2);
plot(outA.tvec, outA.R, 'LineWidth',2);
xlabel('Time'); ylabel('R(t) (radius of gyration)');
legend('baseline','adhesion','Location','best');
grid on; title('Population radius vs time');
drawnow;

%% --------------------------- SIMULATION (S2) ----------------------------
P2 = P;
P2.Ncells = 80;
P2.T  = 1.0;
P2.dt = 1e-3;
P2.Nt = round(P2.T/P2.dt);
P2.mu = 0.0;           % pure motility statistics
P2.ecm_grad = [0;0];   % no drift

nF = numel(P2.F_test);
nRep = P2.nRep;

MSD_baseline = zeros(nF, nRep);
MSD_adhesion = zeros(nF, nRep);


% optional: make P2 available to workers cheaply
P2const = parallel.pool.Constant(P2);

for iF = 1:nF
    F = P2.F_test(iF);

    % preallocate per-iteration cell arrays for parfor (if you need to store full outputs)
    % Here we only need MSD values, so write directly into sliced arrays.
    parfor r = 1:nRep
        % reproducible independent streams:
        rng(1000 + 100*iF + r, 'combRecursive');

        P_local = P2const.Value;
        X0 = initialise_cells(P_local);

        outB = simulate_MCC(P_local, X0, 'baseline', [], F);
        outA = simulate_MCC(P_local, X0, 'adhesion',  [], F);

        % compute MSD at final time (mean over cells)
        msdB = mean(sum((outB.X - X0).^2, 2));
        msdA = mean(sum((outA.X - X0).^2, 2));

        MSD_baseline(iF, r) = msdB;
        MSD_adhesion(iF, r) = msdA;
    end
end

clear P2const

figure('Color','w'); hold on
errorbar(P2.F_test, mean(MSD_baseline,2), std(MSD_baseline,[],2), '-o', 'LineWidth', 1.5);
errorbar(P2.F_test, mean(MSD_adhesion,2), std(MSD_adhesion,[],2), '-o', 'LineWidth', 1.5);
xlabel('Force F (pN)');
ylabel('MSD at T (mean over cells)');
legend('baseline','adhesion','Location','best');
title('Motility statistics (MSD) vs force');
grid on; box on;
drawnow;


%% =============================== FUNCTIONS ===============================
%% =============================== FUNCTIONS ===============================
%% =============================== FUNCTIONS ===============================

%% ------------------------------------------
function out = simulate_MCC(P, X0, mode, snapTimes, F_override, recordR)

if nargin < 3 || isempty(mode),        mode = 'baseline'; end
if nargin < 4 || isempty(snapTimes),   snapTimes = [];         end
if nargin < 5 || isempty(F_override),  F_override = [];        end
if nargin < 6 || isempty(recordR),     recordR = false;        end

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
else
    R = [];
    tvec = [];
end
% time loop (per timestep)
for t = 1:P.Nt
    % ECM and drift
    gradv = ecm_gradient(P, X);
    X = X + (P.mu * gradv) * P.dt;

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

    % ---------- (nested)
    function gradv = ecm_gradient(P, X)
    % ECM gradient field. For clean interpretation we keep it simple.
        N = size(X,1);
        g = P.ecm_grad(:);
        gradv = repmat(g(:).', N, 1);
    end
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
% Sum of obstruction fractions gamma_{k,i} over neighbours within interaction radius.
% Here we use a simple overlap proxy based on distance; replace with a sharper geometric model if desired.

N = size(X,1);
gammaSum = zeros(N,1);

for k = 1:N
    dx = X(:,1) - X(k,1);
    dy = X(:,2) - X(k,2);
    d  = hypot(dx,dy);

    nbr = find(d > 0 & d < P.Rint);
    if isempty(nbr), continue; end

    % Proxy for shared surface area: decreases linearly with distance
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
% Compute per-cell multiplicative reduction factor for jump magnitude based on
% force-dependent stochastic bond lifetimes.
%
% We implement the spirit of the draft:
%   - sample mean lifetimes of catch/slip bonds (Gamma),
%   - combine into alpha_{k,i} = nc*Xbar_c + ns*Xbar_s,
%   - scale jumps with kappa / alpha, aggregated over bonded neighbours.

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

    % For simplicity: treat all neighbours within Rint as bonded (D_{k,i}=1).
    % If you later implement explicit bond formation/breaking, replace this.
    alphaSumInv = 0;

    for j = nbr(:).'
        % Catch-slip switch time: we use a simple instantaneous mapping:
        % - If F below switch, mixture possible; above switch, mostly slip.
        % You can replace this with the px,ss(F,t) logic if you explicitly track contact time.

        Fj = F(k); % force used for bonds of cell k (per-cell proxy)

        % Sample mean bond lifetimes (per bond-type) using Gamma with prescribed mean/SD
        [Lc, sigc] = catch_stats(P, Fj);
        [Ls, sigs] = slip_stats(P, Fj, 0.3); % choose representative contact time for slip

        % Approximate mean of many bonds: Xbar ~ Normal(L, sig^2/n)
        Xbar_c = max(1e-6, Lc + sqrt(sigc^2/max(P.ncatch,1))*randn());
        Xbar_s = max(1e-6, Ls + sqrt(sigs^2/max(P.nslip,1))*randn());

        alpha = P.ncatch * Xbar_c + P.nslip * Xbar_s; % total lifetime proxy

        alphaSumInv = alphaSumInv + (P.kappa / alpha);
    end

    % Reduction factor: more/longer bonds -> smaller jumps
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
function [Ls, sigs] = slip_stats(P, F, tau)
% Mean and SD for slip bonds as functions of force and contact time (the fitted forms).
% tau in seconds.

% From the manuscript (rounded values)
Ls = 0.127 * exp(0.986*tau - 0.102*F) / (1 + 0.527*exp(0.986*tau));
sigs = 0.611*Ls + 0.0113;

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

%% ----------------------------------
function plot_snapshots(outB, outA, P, snapTimes)
% Visual comparison of snapshots: baseline vs adhesion.

if isempty(outB.snapshots) || isempty(outA.snapshots), return; end

figure('Color','w');
nS = numel(outB.snapshots);
for i = 1:nS
    XB = outB.snapshots{i};
    XA = outA.snapshots{i};

    subplot(2,nS,i);
    plot(XB(:,1), XB(:,2), '.', 'MarkerSize',6);
    axis([0 P.Lx 0 P.Ly]); axis square;
    title(sprintf('Baseline t=%.2f', snapTimes(i)));
    set(gca,'XTick',[],'YTick',[]);

    subplot(2,nS,nS+i);
    plot(XA(:,1), XA(:,2), '.', 'MarkerSize',6);
    axis([0 P.Lx 0 P.Ly]); axis square;
    title(sprintf('Adhesion t=%.2f', snapTimes(i)));
    set(gca,'XTick',[],'YTick',[]);
end
end

