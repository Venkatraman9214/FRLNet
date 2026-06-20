% frlnet_policy_manager_sim.m
% Self-contained MATLAB prototype for the cleaned-up FRLNet Policy Manager.
%
% Key design choices reflected here:
%   1) Discrete-time queueing system
%   2) Two queues (extendable to N queues)
%   3) V(t) is the backlog vector, Vmax(t) = max_i V_i(t)
%   4) Exploration = model-free sampling
%   5) Exploitation = model-based value iteration on estimated truncated MDP
%   6) Episodes are event-driven (not fixed-length)
%   7) No central server; decisions are local to the simulated policy manager
%
% NOTE:
% This is a practical simulation prototype, not the full theorem proof.

clear; close all; clc;
rng(1); % Reproducibility

% -------------------------------------------------------------------------
% 1) Simulation parameters
% -------------------------------------------------------------------------

% Queue arrival rates (packets/slot)
lambda = [0.35, 0.25];

% True service success probabilities for the two queues
% (These are only used by the simulator to generate transitions.)
piTrue = [0.60, 0.85];

% Queue truncation threshold (packets)
% This defines the truncated MDP state space: q_i in {0,1,...,T}.
T = 10;

% Practical approximation for S_in.
% In the paper, S_in is defined analytically using the Lyapunov function.
% In the code, we use a simple low-backlog surrogate:
%   S_in approx { V : Vmax <= T_in }
gammaBound = 1;
T_in = max(T - gammaBound, 0);

% Policy-manager hyperparameters
r = 0.5;        % exploration-decay parameter
R = 10;         % episode-transition threshold base
discountFactor = 0.98;  % discount used in value iteration
valueTol = 1e-4;        % stopping tolerance for value iteration
maxVIters = 500;        % maximum value-iteration iterations
laplacePrior = 1;       % smoothing constant for transition estimates

% Simulation horizon
totalSlotsTarget = 10000;

% Maximum number of episodes (safety cap)
maxEpisodes = 500;

% -------------------------------------------------------------------------
% 2) State space and data structures
% -------------------------------------------------------------------------

% Number of states in the truncated 2-queue model
% q1 = 0...T and q2 = 0...T  =>  (T+1)^2 states
numStates = (T + 1)^2;
numActions = 2; % action 1 = serve queue 1, action 2 = serve queue 2

% Transition-count tensor:
% transCounts(nextState, currentState, action)
% Initialize with ones for Laplace smoothing / nonzero prior.
transCounts = ones(numStates, numStates, numActions) * laplacePrior;

% Diagnostic counters
visitCounts = zeros(numStates, numActions);

% Runtime traces
totalBacklogTrace = zeros(totalSlotsTarget, 1);
episodeTrace      = zeros(totalSlotsTarget, 1);
phaseTrace        = zeros(totalSlotsTarget, 1); % 1 = explore, 2 = exploit
actionTrace       = zeros(totalSlotsTarget, 1);

% Initial queue backlog state V(t) = [Q1(t); Q2(t)]
V = [0; 0];

% Episode counter (theta in the manuscript)
episode = 1;

% Time-slot counter
timeSlot = 0;

% Store episode lengths
episodeLengths = [];

% -------------------------------------------------------------------------
% 3) Main learning loop
% -------------------------------------------------------------------------
while timeSlot < totalSlotsTarget && episode <= maxEpisodes

    % ---------------------------------------------------------------------
    % 3a) Episode-level policy selection
    % ---------------------------------------------------------------------
    % mu is a random draw in [0,1] used to select exploration or exploitation
    % for the current episode.
    mu = rand;

    % Exploration threshold for this episode
    zeta_theta = r / sqrt(episode);

    % Episode transition threshold
    R_theta = R * sqrt(episode);

    % Decide the phase for the entire episode
    isExplorationEpisode = (mu <= zeta_theta);

    % ---------------------------------------------------------------------
    % 3b) Estimate the truncated MDP from the data collected so far
    % ---------------------------------------------------------------------
    % Build the empirical transition model P_hat from transition counts.
    P_hat = estimateTransitionModel(transCounts);

    % Cost vector: C_f(V) = Q1 + Q2 (aggregate backlog)
    costVec = buildCostVector(T);

    % Solve the estimated truncated MDP via value iteration.
    % policyStar(s) returns action 1 or 2 for each state s.
    [~, policyStar] = valueIterationMDP( ...
        P_hat, costVec, discountFactor, valueTol, maxVIters);

    % ---------------------------------------------------------------------
    % 3c) Start a new episode
    % ---------------------------------------------------------------------
    episodeSlotCounter = 0;
    episodeVisitsIn = 0;
    episodeClosed = false;

    while timeSlot < totalSlotsTarget

        timeSlot = timeSlot + 1;
        episodeSlotCounter = episodeSlotCounter + 1;

        % Current backlog components (discrete time)
        q1 = V(1);
        q2 = V(2);

        % Current state index in the truncated MDP
        sIdx = state2idx(q1, q2, T);

        % Vmax(t) = max_i V_i(t)
        Vmax = max(V);

        % Practical surrogate for S_in:
        % states in the low-backlog region are treated as "inside"
        isIn = (Vmax <= T_in);

        % -----------------------------------------------------------------
        % 3d) Choose action according to the current episode phase
        % -----------------------------------------------------------------
        if isExplorationEpisode
            % Model-free exploration:
            % choose a random action inside the truncated region.
            % Outside S_in, fall back to the baseline stabilizing policy.
            if isIn
                action = randi(numActions);
            else
                action = baselinePolicy(V);
            end
            currentPhase = 1;
        else
            % Model-based exploitation:
            % use the learned policy inside S_in; outside S_in use beta_0.
            if isIn
                action = policyStar(sIdx);
            else
                action = baselinePolicy(V);
            end
            currentPhase = 2;
        end

        % -----------------------------------------------------------------
        % 3e) Apply one-step queue dynamics
        % -----------------------------------------------------------------
        % The environment simulator generates the next state based on:
        %   - current queue lengths
        %   - chosen action
        %   - Bernoulli arrivals
        %   - probabilistic transmission success
        Vnext = simulateQueueStep(V, action, lambda, piTrue, T);

        % -----------------------------------------------------------------
        % 3f) Update empirical transition model
        % -----------------------------------------------------------------
        nextIdx = state2idx(Vnext(1), Vnext(2), T);
        transCounts(nextIdx, sIdx, action) = transCounts(nextIdx, sIdx, action) + 1;
        visitCounts(sIdx, action) = visitCounts(sIdx, action) + 1;

        % -----------------------------------------------------------------
        % 3g) Log diagnostics
        % -----------------------------------------------------------------
        totalBacklogTrace(timeSlot) = sum(V);   % aggregate backlog
        episodeTrace(timeSlot)      = episode;
        phaseTrace(timeSlot)        = currentPhase;
        actionTrace(timeSlot)       = action;

        % Count visits to the practical S_in region for episode termination
        if isIn
            episodeVisitsIn = episodeVisitsIn + 1;
        end

        % Advance the state
        V = Vnext;

        % -----------------------------------------------------------------
        % 3h) Episode termination condition
        % -----------------------------------------------------------------
        % The episode ends once the learning threshold is reached.
        % In this practical prototype, the threshold is implemented as
        % a minimum number of visits to the inner region S_in.
        if episodeVisitsIn >= R_theta
            episodeLengths(end + 1) = episodeSlotCounter; %#ok<SAGROW>
            episodeClosed = true;
            break;
        end
    end

    % If the simulation horizon ended before the episode closed,
    % still record the partial episode length.
    if ~episodeClosed && episodeSlotCounter > 0
        episodeLengths(end + 1) = episodeSlotCounter; %#ok<SAGROW>
    end

    % Move to the next episode
    episode = episode + 1;
end

% Trim traces to the actual number of simulated slots
validIdx = 1:timeSlot;
totalBacklogTrace = totalBacklogTrace(validIdx);
episodeTrace      = episodeTrace(validIdx);
phaseTrace        = phaseTrace(validIdx);
actionTrace       = actionTrace(validIdx);

% -------------------------------------------------------------------------
% 4) Plots
% -------------------------------------------------------------------------

% Aggregate backlog over time
figure;
plot(validIdx, totalBacklogTrace, 'LineWidth', 1.8);
xlabel('Time slot', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Aggregate backlog \Sigma_i Q_i', 'FontSize', 14, 'FontWeight', 'bold');
title('FRLNet Policy Manager: Aggregate Backlog Over Time');
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_backlog_trace.png', 'Resolution', 600);

% Episode lengths
figure;
plot(episodeLengths, '-o', 'LineWidth', 1.8);
xlabel('Episode index', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Episode length (slots)', 'FontSize', 14, 'FontWeight', 'bold');
title('Event-Driven Episode Lengths');
grid on;
ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';
set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
exportgraphics(gcf, 'frlnet_episode_lengths.png', 'Resolution', 600);

% -------------------------------------------------------------------------
% 5) Console summary
% -------------------------------------------------------------------------
fprintf('Simulation finished.\n');
fprintf('  Total slots simulated : %d\n', timeSlot);
fprintf('  Episodes completed    : %d\n', episode - 1);
fprintf('  Mean aggregate backlog: %.3f packets\n', mean(totalBacklogTrace));
fprintf('  Final state            : [Q1,Q2] = [%d,%d]\n', V(1), V(2));

% =========================================================================
% Local helper functions (keep them at the end of the script)
% =========================================================================

function idx = state2idx(q1, q2, T)
    % Maps the pair (q1,q2) with q1,q2 in {0,...,T} to a unique state index.
    q1 = min(max(round(q1), 0), T);
    q2 = min(max(round(q2), 0), T);
    idx = sub2ind([T + 1, T + 1], q1 + 1, q2 + 1);
end

function Vnext = simulateQueueStep(V, action, lambda, piTrue, T)
    % Simulate one discrete-time queue update.
    %
    % Order of events:
    %   1) Serve one queue according to the selected action.
    %   2) Apply Bernoulli arrivals.
    %   3) Truncate queue lengths at T.

    q1 = V(1);
    q2 = V(2);

    % -----------------------------
    % Service step
    % -----------------------------
    if action == 1 && q1 > 0
        % Serve queue 1
        if rand < piTrue(1)
            q1 = q1 - 1;
        end
    elseif action == 2 && q2 > 0
        % Serve queue 2
        if rand < piTrue(2)
            q2 = q2 - 1;
        end
    end

    % -----------------------------
    % Arrival step
    % -----------------------------
    % Bernoulli arrivals: 0 or 1 packet per slot per queue
    q1 = q1 + (rand < lambda(1));
    q2 = q2 + (rand < lambda(2));

    % -----------------------------
    % Truncation step
    % -----------------------------
    q1 = min(q1, T);
    q2 = min(q2, T);

    Vnext = [q1; q2];
end

function action = baselinePolicy(V)
    % Baseline stabilizing policy beta_0:
    % longest-queue-first (LQF) with a deterministic tie-breaker.
    %
    % If the queues are equal, we choose queue 1 by default.

    if V(1) > V(2)
        action = 1;
    elseif V(2) > V(1)
        action = 2;
    else
        action = 1; % tie-breaker
    end
end

function costVec = buildCostVector(T)
    % State cost C_f(V) = Q1 + Q2.
    % This is the aggregate backlog objective used in the script.

    numStates = (T + 1)^2;
    costVec = zeros(numStates, 1);

    for q1 = 0:T
        for q2 = 0:T
            idx = sub2ind([T + 1, T + 1], q1 + 1, q2 + 1);
            costVec(idx) = q1 + q2;
        end
    end
end

function P_hat = estimateTransitionModel(transCounts)
    % Convert transition counts into a smoothed transition model.
    %
    % P_hat(nextState, currentState, action)
    %
    % Laplace smoothing avoids zero-probability rows and makes the
    % estimated MDP well-posed even early in learning.

    [S, ~, A] = size(transCounts);
    P_hat = zeros(S, S, A);

    for a = 1:A
        for s = 1:S
            c = transCounts(:, s, a);
            c = c + 1; % Laplace smoothing
            P_hat(:, s, a) = c / sum(c);
        end
    end
end

function [V, policy] = valueIterationMDP(P_hat, costVec, discountFactor, tol, maxIter)
    % Solve the estimated truncated MDP using value iteration.
    %
    % Returns:
    %   V      - optimal value function
    %   policy - optimal action for each state
    %
    % This is the model-based part of the policy manager.

    [S, ~, A] = size(P_hat);
    V = zeros(S, 1);
    policy = ones(S, 1);

    for iter = 1:maxIter
        Vnew = zeros(S, 1);

        for s = 1:S
            Q = zeros(A, 1);
            for a = 1:A
                % One-step cost + discounted expected next value
                Q(a) = costVec(s) + discountFactor * (P_hat(:, s, a)' * V);
            end

            [Vnew(s), policy(s)] = min(Q);
        end

        if max(abs(Vnew - V)) < tol
            V = Vnew;
            break;
        end

        V = Vnew;
    end
end