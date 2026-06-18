% EEE 448 - Project Report II: Tasks 3 and 4
% Dynamic Programming, Model Learning, Q-learning, and Tabular Policy Gradient
% Cagan Izol - 22303351

clear; clc; close all;
rng(448);

%% ------------------------- Parameters -------------------------
gamma = 0.95;
x_bar = 20;
Bmax = 50;
mmax = 10;

x0 = 10;
B0 = 30;
m0 = 0;
mu0 = 2;        % 1=Bear, 2=Sideways, 3=Bull

P_M = [0.5 0.5 0.0;
       0.2 0.5 0.3;
       0.0 0.4 0.6];

omega_vals = [-10 -5 -1 1 5 10];

P_noise = [0.1 0.2 0.4 0.2 0.1 0.0;
           0.0 0.1 0.4 0.4 0.1 0.0;
           0.0 0.1 0.2 0.4 0.2 0.1];

nx = x_bar + 1;
nB = Bmax + 1;
nm = mmax + 1;
nmu = 3;
N = nx*nB*nm*nmu;
maxA = 11;

%% ------------------------- Build state/action tables -------------------------
[xArr, BArr, mArr, muArr, numActs, actMat, BplusMat, mplusMat, holdIdx] = ...
    build_state_action_tables(x_bar, Bmax, mmax, maxA);

s0 = state_id(x0, B0, m0, mu0, Bmax, mmax);

fprintf('Total number of states: %d\n', N);
fprintf('Initial state id: %d\n', s0);

%% ------------------------- Task 3.1.1: Value Iteration -------------------------
tolVI = 1e-8;
maxIterVI = 3000;

[V_VI, pi_VI, VI_deltas] = value_iteration_known( ...
    P_M, P_noise, omega_vals, gamma, x_bar, Bmax, mmax, ...
    xArr, BArr, mArr, muArr, numActs, actMat, BplusMat, mplusMat, ...
    tolVI, maxIterVI);

fprintf('\n--- Value Iteration ---\n');
fprintf('Sweeps: %d\n', length(VI_deltas));
fprintf('Final Bellman residual: %.3e\n', VI_deltas(end));
fprintf('V*(s0) = %.6f\n', V_VI(s0));
fprintf('a*(s0) = %d\n', actMat(s0, pi_VI(s0)));

%% ------------------------- Task 3.1.2: Policy Iteration -------------------------
tolEval = 1e-10;
maxEvalIter = 3000;
maxPISweeps = 100;

[V_PI, pi_PI, PI_info] = policy_iteration_known( ...
    P_M, P_noise, omega_vals, gamma, x_bar, Bmax, mmax, ...
    xArr, BArr, mArr, muArr, numActs, actMat, BplusMat, mplusMat, holdIdx, ...
    tolEval, maxEvalIter, maxPISweeps);

fprintf('\n--- Policy Iteration ---\n');
fprintf('Policy improvement sweeps: %d\n', size(PI_info,1));
fprintf('V^pi(s0) = %.6f\n', V_PI(s0));
fprintf('a_PI(s0) = %d\n', actMat(s0, pi_PI(s0)));
fprintf('Policy agreement with VI: %.2f%%\n', 100*mean(pi_PI == pi_VI));

%% ------------------------- Task 3.2: Simulate DP policy -------------------------
numTrials = 100000;
T = 2000;

[retDP, finalWDP, bankruptDP] = simulate_policy( ...
    pi_VI, numTrials, T, gamma, x0, B0, m0, mu0, ...
    P_M, P_noise, omega_vals, x_bar, Bmax, mmax, actMat);

fprintf('\n--- DP Policy Simulation ---\n');
fprintf('Expected V*(s0): %.6f\n', V_VI(s0));
fprintf('Empirical return: %.6f +/- %.6f (95%% CI)\n', mean(retDP), 1.96*std(retDP)/sqrt(numTrials));
fprintf('Bankruptcy rate: %.3f%%\n', 100*mean(bankruptDP));
fprintf('Mean final wealth: %.6f\n', mean(finalWDP));

%% ------------------------- Task 3.3: Learn model, repeat DP -------------------------
sampleTrajList = [20 100 500 2000];
trajLen = 100;
numLearnedEvalTrials = 20000;


learnedSamples = sampleTrajList(:) * trajLen;
learnedVhat = zeros(length(sampleTrajList),1);
learnedVtrue = zeros(length(sampleTrajList),1);
learnedSimReturn = zeros(length(sampleTrajList),1);
learnedFinalWealth = zeros(length(sampleTrajList),1);
learnedBankruptcy = zeros(length(sampleTrajList),1);
learnedPolicies = cell(length(sampleTrajList),1);

fprintf('\n--- Learned Model Dynamic Programming ---\n');
for i = 1:length(sampleTrajList)
    nTraj = sampleTrajList(i);
    K_hat = learn_exogenous_kernel(nTraj, trajLen, P_M, P_noise, omega_vals, x_bar);
    [V_L, pi_L, ~] = value_iteration_kernel( ...
        K_hat, gamma, x_bar, Bmax, mmax, ...
        xArr, BArr, mArr, muArr, numActs, actMat, BplusMat, mplusMat, ...
        tolVI, maxIterVI);
    [V_eval_L, ~] = evaluate_policy_known(pi_L, ...
        P_M, P_noise, omega_vals, gamma, x_bar, Bmax, mmax, ...
        xArr, BArr, mArr, muArr, actMat, BplusMat, mplusMat, ...
        tolEval, maxEvalIter);
    [retL, finalWL, bankruptL] = simulate_policy(pi_L, numLearnedEvalTrials, T, gamma, x0, B0, m0, mu0, ...
        P_M, P_noise, omega_vals, x_bar, Bmax, mmax, actMat);

    learnedVhat(i) = V_L(s0);
    learnedVtrue(i) = V_eval_L(s0);
    learnedSimReturn(i) = mean(retL);
    learnedFinalWealth(i) = mean(finalWL);
    learnedBankruptcy(i) = mean(bankruptL);
    learnedPolicies{i} = pi_L;

    fprintf('Samples=%7d | learned V=%.4f | true V=%.4f | sim=%.4f | final wealth=%.4f | bankrupt=%.2f%%\n', ...
        learnedSamples(i), learnedVhat(i), learnedVtrue(i), learnedSimReturn(i), learnedFinalWealth(i), 100*learnedBankruptcy(i));
end


bestLearnedIdx = length(sampleTrajList);
pi_L_best = learnedPolicies{bestLearnedIdx};
V_eval_L_best_s0 = learnedVtrue(bestLearnedIdx);
retLearnedBest = learnedSimReturn(bestLearnedIdx);
finalWLearnedBest = learnedFinalWealth(bestLearnedIdx);
bankruptLearnedBest = learnedBankruptcy(bestLearnedIdx);

LearnedModelTable = table(learnedSamples, learnedVhat, learnedVtrue, learnedSimReturn, learnedFinalWealth, 100*learnedBankruptcy, ...
    'VariableNames', {'SampleTransitions','LearnedModelValue','TrueExpectedValue','EmpiricalDiscountedReturn','MeanFinalWealth','BankruptcyPercent'});
fprintf('\nTask 3.3 Learned-Model Comparison Table:\n');
disp(LearnedModelTable);

%% ------------------------- Task 4.1.1: Q-learning -------------------------
episodesQ = 2000000;
horizonQ = 250;
eps0 = 0.80; epsEnd = 0.01;
alpha0 = 0.08; alphaEnd = 0.005;
randomStartProb = 0.20;

[Q, pi_Q, trainRetQ] = q_learning(episodesQ, horizonQ, gamma, ...
    eps0, epsEnd, alpha0, alphaEnd, randomStartProb, ...
    P_M, P_noise, omega_vals, x_bar, Bmax, mmax, ...
    numActs, actMat, BplusMat, mplusMat, holdIdx);

[V_eval_Q, ~] = evaluate_policy_known(pi_Q, ...
    P_M, P_noise, omega_vals, gamma, x_bar, Bmax, mmax, ...
    xArr, BArr, mArr, muArr, actMat, BplusMat, mplusMat, tolEval, maxEvalIter);

[retQ, finalWQ, bankruptQ] = simulate_policy(pi_Q, numTrials, T, gamma, x0, B0, m0, mu0, ...
    P_M, P_noise, omega_vals, x_bar, Bmax, mmax, actMat);

fprintf('\n--- Q-learning ---\n');
fprintf('Expected value of learned policy: %.6f\n', V_eval_Q(s0));
fprintf('Empirical return: %.6f +/- %.6f (95%% CI)\n', mean(retQ), 1.96*std(retQ)/sqrt(numTrials));
fprintf('Bankruptcy rate: %.3f%%\n', 100*mean(bankruptQ));
fprintf('Mean final wealth: %.6f\n', mean(finalWQ));

%% ------------------------- Task 4.1.2: Tabular Policy Gradient / Actor-Critic -------------------------
episodesPG = 2000000;
horizonPG = 250;
thetaLR0 = 0.0015; thetaLREnd = 0.00005;
vLR0 = 0.03; vLREnd = 0.001;
temp0 = 1.0; tempEnd = 0.1;
clipDelta = 100;

[theta, Vcritic, pi_PG, trainRetPG] = actor_critic_pg(episodesPG, horizonPG, gamma, ...
    thetaLR0, thetaLREnd, vLR0, vLREnd, temp0, tempEnd, clipDelta, ...
    P_M, P_noise, omega_vals, x_bar, Bmax, mmax, ...
    numActs, actMat, BplusMat, mplusMat, holdIdx);

[V_eval_PG, ~] = evaluate_policy_known(pi_PG, ...
    P_M, P_noise, omega_vals, gamma, x_bar, Bmax, mmax, ...
    xArr, BArr, mArr, muArr, actMat, BplusMat, mplusMat, tolEval, maxEvalIter);

[retPG, finalWPG, bankruptPG] = simulate_policy(pi_PG, numTrials, T, gamma, x0, B0, m0, mu0, ...
    P_M, P_noise, omega_vals, x_bar, Bmax, mmax, actMat);

fprintf('\n--- Policy Gradient / Actor-Critic ---\n');
fprintf('Expected value of learned policy: %.6f\n', V_eval_PG(s0));
fprintf('Empirical return: %.6f +/- %.6f (95%% CI)\n', mean(retPG), 1.96*std(retPG)/sqrt(numTrials));
fprintf('Bankruptcy rate: %.3f%%\n', 100*mean(bankruptPG));
fprintf('Mean final wealth: %.6f\n', mean(finalWPG));

%% ------------------------- Summary and Comparison Tables -------------------------
fprintf('\n================ Final Summary ================\n');
fprintf('DP optimal:       expected %.4f, sim %.4f, bankruptcy %.3f%%\n', V_VI(s0), mean(retDP), 100*mean(bankruptDP));
fprintf('Learned model DP: expected %.4f, sim %.4f, bankruptcy %.3f%%\n', V_eval_L_best_s0, retLearnedBest, 100*bankruptLearnedBest);
fprintf('Q-learning:       expected %.4f, sim %.4f, bankruptcy %.3f%%\n', V_eval_Q(s0), mean(retQ), 100*mean(bankruptQ));
fprintf('Actor-Critic PG:  expected %.4f, sim %.4f, bankruptcy %.3f%%\n', V_eval_PG(s0), mean(retPG), 100*mean(bankruptPG));

Method = {'Known-model DP'; 'Learned-model DP'; 'Q-learning'; 'Actor-Critic PG'};
ExpectedValue = [V_VI(s0); V_eval_L_best_s0; V_eval_Q(s0); V_eval_PG(s0)];
EmpiricalDiscountedReturn = [mean(retDP); retLearnedBest; mean(retQ); mean(retPG)];
BankruptcyPercent = 100*[mean(bankruptDP); bankruptLearnedBest; mean(bankruptQ); mean(bankruptPG)];
PerformanceComparisonTable = table(Method, ExpectedValue, EmpiricalDiscountedReturn, BankruptcyPercent);

fprintf('\nOverall Performance Comparison Table:\n');
disp(PerformanceComparisonTable);

InitialWealth = (B0 + m0*x0) * ones(4,1);
MeanFinalWealth = [mean(finalWDP); finalWLearnedBest; mean(finalWQ); mean(finalWPG)];
MeanWealthChange = MeanFinalWealth - InitialWealth;
AverageWealthTable = table(Method, InitialWealth, MeanFinalWealth, MeanWealthChange);

fprintf('\nAverage Wealth Comparison Table:\n');
disp(AverageWealthTable);

%% ------------------------- Figure and Table Export -------------------------
figDir = 'figures_task3_task4';
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

figure;
semilogy(1:length(VI_deltas), VI_deltas, 'LineWidth', 1.5);
grid on;
xlabel('Value Iteration Sweep');
ylabel('Bellman Residual');
title('Value Iteration Convergence');
saveas(gcf, fullfile(figDir, 'fig1_value_iteration_convergence.png'));

figure;
plot(PI_info(:,1), PI_info(:,3), '-o', 'LineWidth', 1.5);
grid on;
xlabel('Policy Iteration Sweep');
ylabel('Number of Changed States');
title('Policy Iteration Convergence');
saveas(gcf, fullfile(figDir, 'fig2_policy_iteration_convergence.png'));

figure;
bar(categorical(Method), EmpiricalDiscountedReturn);
ylabel('Empirical Discounted Return');
title('Performance Comparison Including Learned Model');
grid on;
saveas(gcf, fullfile(figDir, 'fig3_performance_comparison_with_learned_model.png'));

figure;
bar(categorical(Method), BankruptcyPercent);
ylabel('Bankruptcy Rate (%)');
title('Bankruptcy Comparison Including Learned Model');
grid on;
saveas(gcf, fullfile(figDir, 'fig4_bankruptcy_comparison_with_learned_model.png'));

figure;
plot(learnedSamples, learnedVtrue, '-o', 'LineWidth', 1.5);
hold on;
plot(learnedSamples, learnedSimReturn, '-s', 'LineWidth', 1.5);
grid on;
xlabel('Number of Sample Transitions Used to Learn Model');
ylabel('Performance');
legend('True expected value of learned policy','Empirical discounted return','Location','best');
title('Effect of Sample Size on Learned-Model Performance');
saveas(gcf, fullfile(figDir, 'fig5_learned_model_sample_size.png'));

figure;
bar(categorical(Method), MeanFinalWealth);
ylabel('Mean Final Wealth');
title('Average Wealth Across Models');
grid on;
saveas(gcf, fullfile(figDir, 'fig6_average_wealth_across_models.png'));

figure;
plot(movmean(trainRetQ,5000),'LineWidth',1.2);
hold on;
plot(movmean(trainRetPG,5000),'LineWidth',1.2);
grid on;
xlabel('Episode');
ylabel('Moving Average Return');
legend('Q-learning','Actor-Critic');
title('Training Curves');
saveas(gcf, fullfile(figDir, 'fig7_training_curves.png'));

writetable(LearnedModelTable, fullfile(figDir, 'table_learned_model_comparison.csv'));
writetable(PerformanceComparisonTable, fullfile(figDir, 'table_overall_performance_comparison.csv'));
writetable(AverageWealthTable, fullfile(figDir, 'table_average_wealth_comparison.csv'));

fprintf('\nFigures and CSV tables saved in folder: %s\n', figDir);

%% ========================================================================
%                               FUNCTIONS
% ========================================================================

function id = state_id(x, B, m, mu, Bmax, mmax)
    % x,B,m are zero-based values; mu is 1,2,3.
    id = (((x*(Bmax+1) + B)*(mmax+1) + m)*3 + mu);
end

function [x, B, m, mu] = decode_state(id, Bmax, mmax)
    z = id - 1;
    mu = mod(z,3) + 1;
    z = floor(z/3);
    m = mod(z,mmax+1);
    z = floor(z/(mmax+1));
    B = mod(z,Bmax+1);
    x = floor(z/(Bmax+1));
end

function xp = price_update(x, omega, x_bar)
    if x + omega >= x_bar
        xp = x_bar;
    elseif x > 0 && x + omega >= 0
        xp = x + omega;
    else
        xp = 0;
    end
end

function [xArr, BArr, mArr, muArr, numActs, actMat, BplusMat, mplusMat, holdIdx] = ...
    build_state_action_tables(x_bar, Bmax, mmax, maxA)
    N = (x_bar+1)*(Bmax+1)*(mmax+1)*3;
    xArr = zeros(N,1); BArr = zeros(N,1); mArr = zeros(N,1); muArr = zeros(N,1);
    numActs = zeros(N,1);
    actMat = zeros(N,maxA);
    BplusMat = zeros(N,maxA);
    mplusMat = zeros(N,maxA);
    holdIdx = ones(N,1);
    for x = 0:x_bar
        for B = 0:Bmax
            for m = 0:mmax
                for mu = 1:3
                    s = state_id(x,B,m,mu,Bmax,mmax);
                    xArr(s)=x; BArr(s)=B; mArr(s)=m; muArr(s)=mu;
                    if x > 0
                        sellMax = min(m, floor((Bmax-B)/x));
                        buyMax = min(floor(B/x), mmax-m);
                        A = -sellMax:buyMax;
                    else
                        A = -m:0;
                    end
                    numActs(s) = length(A);
                    for ai = 1:length(A)
                        a = A(ai);
                        actMat(s,ai) = a;
                        BplusMat(s,ai) = B - a*x;
                        mplusMat(s,ai) = m + a;
                        if a == 0
                            holdIdx(s) = ai;
                        end
                    end
                end
            end
        end
    end
end

function [V, policy, deltas] = value_iteration_known(P_M, P_noise, omega_vals, gamma, x_bar, Bmax, mmax, xArr, BArr, mArr, muArr, numActs, actMat, BplusMat, mplusMat, tol, maxIter)
    N = length(xArr);
    V = zeros(N,1);
    policy = ones(N,1);
    deltas = zeros(maxIter,1);
    for it = 1:maxIter
        Vnew = zeros(N,1);
        delta = 0;
        for s = 1:N
            x = xArr(s); mu = muArr(s);
            bestQ = -Inf; bestAI = 1;
            for ai = 1:numActs(s)
                Bp = BplusMat(s,ai);
                mp = mplusMat(s,ai);
                q = 0;
                for wi = 1:length(omega_vals)
                    pw = P_noise(mu,wi);
                    if pw == 0, continue; end
                    xp = price_update(x, omega_vals(wi), x_bar);
                    r = mp*(xp - x);
                    for mup = 1:3
                        p = pw*P_M(mu,mup);
                        if p == 0, continue; end
                        sp = state_id(xp,Bp,mp,mup,Bmax,mmax);
                        q = q + p*(r + gamma*V(sp));
                    end
                end
                if q > bestQ
                    bestQ = q; bestAI = ai;
                end
            end
            Vnew(s) = bestQ;
            policy(s) = bestAI;
            delta = max(delta, abs(Vnew(s)-V(s)));
        end
        V = Vnew;
        deltas(it) = delta;
        if delta < tol
            deltas = deltas(1:it);
            return;
        end
    end
end

function [V, policy, PI_info] = policy_iteration_known(P_M, P_noise, omega_vals, gamma, x_bar, Bmax, mmax, xArr, BArr, mArr, muArr, numActs, actMat, BplusMat, mplusMat, holdIdx, tolEval, maxEvalIter, maxPISweeps)
    N = length(xArr);
    policy = holdIdx;
    PI_info = [];
    for sweep = 1:maxPISweeps
        [V, evalIters] = evaluate_policy_known(policy, P_M, P_noise, omega_vals, gamma, x_bar, Bmax, mmax, xArr, BArr, mArr, muArr, actMat, BplusMat, mplusMat, tolEval, maxEvalIter);
        changes = 0;
        newPolicy = policy;
        for s = 1:N
            x = xArr(s); mu = muArr(s);
            bestQ = -Inf; bestAI = 1;
            for ai = 1:numActs(s)
                Bp = BplusMat(s,ai);
                mp = mplusMat(s,ai);
                q = 0;
                for wi = 1:length(omega_vals)
                    pw = P_noise(mu,wi);
                    if pw == 0, continue; end
                    xp = price_update(x, omega_vals(wi), x_bar);
                    r = mp*(xp - x);
                    for mup = 1:3
                        p = pw*P_M(mu,mup);
                        if p == 0, continue; end
                        sp = state_id(xp,Bp,mp,mup,Bmax,mmax);
                        q = q + p*(r + gamma*V(sp));
                    end
                end
                if q > bestQ
                    bestQ = q; bestAI = ai;
                end
            end
            if bestAI ~= policy(s)
                changes = changes + 1;
                newPolicy(s) = bestAI;
            end
        end
        policy = newPolicy;
        PI_info = [PI_info; sweep evalIters changes V(state_id(10,30,0,2,Bmax,mmax))]; %#ok<AGROW>
        if changes == 0
            return;
        end
    end
end

function [V, evalIters] = evaluate_policy_known(policy, P_M, P_noise, omega_vals, gamma, x_bar, Bmax, mmax, xArr, BArr, mArr, muArr, actMat, BplusMat, mplusMat, tol, maxIter)
    N = length(xArr);
    V = zeros(N,1);
    for it = 1:maxIter
        Vnew = zeros(N,1);
        delta = 0;
        for s = 1:N
            x = xArr(s); mu = muArr(s);
            ai = policy(s);
            Bp = BplusMat(s,ai);
            mp = mplusMat(s,ai);
            val = 0;
            for wi = 1:length(omega_vals)
                pw = P_noise(mu,wi);
                if pw == 0, continue; end
                xp = price_update(x, omega_vals(wi), x_bar);
                r = mp*(xp - x);
                for mup = 1:3
                    p = pw*P_M(mu,mup);
                    if p == 0, continue; end
                    sp = state_id(xp,Bp,mp,mup,Bmax,mmax);
                    val = val + p*(r + gamma*V(sp));
                end
            end
            Vnew(s) = val;
            delta = max(delta, abs(Vnew(s)-V(s)));
        end
        V = Vnew;
        if delta < tol
            evalIters = it;
            return;
        end
    end
    evalIters = maxIter;
end

function [returns, finalWealth, bankrupt] = simulate_policy(policy, numTrials, T, gamma, x0, B0, m0, mu0, P_M, P_noise, omega_vals, x_bar, Bmax, mmax, actMat)
    returns = zeros(numTrials,1);
    finalWealth = zeros(numTrials,1);
    bankrupt = false(numTrials,1);
    for tr = 1:numTrials
        x=x0; B=B0; m=m0; mu=mu0;
        disc = 1; G = 0;
        for k = 1:T
            s = state_id(x,B,m,mu,Bmax,mmax);
            a = actMat(s, policy(s));
            Bp = B - a*x;
            mp = m + a;
            muNext = sample_discrete(P_M(mu,:));
            omega = omega_vals(sample_discrete(P_noise(mu,:)));
            xp = price_update(x, omega, x_bar);
            r = mp*(xp - x);
            G = G + disc*r;
            disc = disc*gamma;
            wealth = Bp + mp*xp;
            if wealth == 0
                bankrupt(tr) = true;
            end
            x=xp; B=Bp; m=mp; mu=muNext;
            if x == 0
                break;
            end
        end
        returns(tr) = G;
        finalWealth(tr) = B + m*x;
    end
end

function idx = sample_discrete(prob)
    u = rand;
    c = cumsum(prob(:));
    idx = find(u <= c, 1, 'first');
end

function K = learn_exogenous_kernel(numTraj, trajLen, P_M, P_noise, omega_vals, x_bar)
    counts = zeros(63,63);
    for tr = 1:numTraj
        x = randi([0 20]);
        mu = randi([1 3]);
        for k = 1:trajLen
            e = x*3 + mu;
            muNext = sample_discrete(P_M(mu,:));
            omega = omega_vals(sample_discrete(P_noise(mu,:)));
            xp = price_update(x, omega, x_bar);
            ep = xp*3 + muNext;
            counts(e,ep) = counts(e,ep) + 1;
            x = xp; mu = muNext;
            if x == 0
                x = randi([0 20]);
                mu = randi([1 3]);
            end
        end
    end
    K = zeros(63,63);
    for e = 1:63
        rowSum = sum(counts(e,:));
        if rowSum > 0
            K(e,:) = counts(e,:)/rowSum;
        else
            K(e,e) = 1; 
        end
    end
end

function [V, policy, deltas] = value_iteration_kernel(K, gamma, x_bar, Bmax, mmax, xArr, BArr, mArr, muArr, numActs, actMat, BplusMat, mplusMat, tol, maxIter)
    N = length(xArr);
    V = zeros(N,1);
    policy = ones(N,1);
    deltas = zeros(maxIter,1);
    for it = 1:maxIter
        Vnew = zeros(N,1);
        delta = 0;
        for s = 1:N
            x = xArr(s); mu = muArr(s);
            e = x*3 + mu;
            nextCols = find(K(e,:) > 0);
            bestQ = -Inf; bestAI = 1;
            for ai = 1:numActs(s)
                Bp = BplusMat(s,ai);
                mp = mplusMat(s,ai);
                q = 0;
                for c = nextCols
                    p = K(e,c);
                    xp = floor((c-1)/3);
                    mup = mod(c-1,3)+1;
                    r = mp*(xp-x);
                    sp = state_id(xp,Bp,mp,mup,Bmax,mmax);
                    q = q + p*(r + gamma*V(sp));
                end
                if q > bestQ
                    bestQ = q; bestAI = ai;
                end
            end
            Vnew(s) = bestQ;
            policy(s) = bestAI;
            delta = max(delta, abs(Vnew(s)-V(s)));
        end
        V = Vnew;
        deltas(it) = delta;
        if delta < tol
            deltas = deltas(1:it);
            return;
        end
    end
end

function [Q, policy, trainReturns] = q_learning(episodes, horizon, gamma, eps0, epsEnd, alpha0, alphaEnd, randomStartProb, P_M, P_noise, omega_vals, x_bar, Bmax, mmax, numActs, actMat, BplusMat, mplusMat, holdIdx)
    N = length(numActs);
    maxA = size(actMat,2);
    Q = zeros(N,maxA);
    visits = zeros(N,maxA);
    trainReturns = zeros(episodes,1);
    for ep = 1:episodes
        frac = (ep-1)/max(1,episodes-1);
        eps = eps0 + frac*(epsEnd-eps0);
        alpha = alpha0 + frac*(alphaEnd-alpha0);
        if rand < randomStartProb
            x = randi([1 x_bar]); B = randi([0 Bmax]); m = randi([0 mmax]); mu = randi([1 3]);
        else
            x = 10; B = 30; m = 0; mu = 2;
        end
        disc = 1; G = 0;
        for k = 1:horizon
            s = state_id(x,B,m,mu,Bmax,mmax);
            if rand < eps
                ai = randi(numActs(s));
            else
                [~, ai] = max(Q(s,1:numActs(s)));
            end
            a = actMat(s,ai);
            Bp = B - a*x;
            mp = m + a;
            muNext = sample_discrete(P_M(mu,:));
            omega = omega_vals(sample_discrete(P_noise(mu,:)));
            xp = price_update(x, omega, x_bar);
            r = mp*(xp-x);
            sp = state_id(xp,Bp,mp,muNext,Bmax,mmax);
            target = r + gamma*max(Q(sp,1:numActs(sp)));
            Q(s,ai) = Q(s,ai) + alpha*(target - Q(s,ai));
            visits(s,ai) = visits(s,ai) + 1;
            G = G + disc*r;
            disc = disc*gamma;
            x=xp; B=Bp; m=mp; mu=muNext;
            if x == 0
                break;
            end
        end
        trainReturns(ep) = G;
    end
    policy = holdIdx;
    for s = 1:N
        if sum(visits(s,1:numActs(s))) > 0
            [~, policy(s)] = max(Q(s,1:numActs(s)));
        end
    end
end

function [theta, Vcritic, policy, trainReturns] = actor_critic_pg(episodes, horizon, gamma, thetaLR0, thetaLREnd, vLR0, vLREnd, temp0, tempEnd, clipDelta, P_M, P_noise, omega_vals, x_bar, Bmax, mmax, numActs, actMat, BplusMat, mplusMat, holdIdx)
    N = length(numActs);
    maxA = size(actMat,2);
    theta = zeros(N,maxA);
    Vcritic = zeros(N,1);
    visits = zeros(N,maxA);
    trainReturns = zeros(episodes,1);
    for ep = 1:episodes
        frac = (ep-1)/max(1,episodes-1);
        thetaLR = thetaLR0 + frac*(thetaLREnd-thetaLR0);
        vLR = vLR0 + frac*(vLREnd-vLR0);
        temp = temp0 + frac*(tempEnd-temp0);
        x = 10; B = 30; m = 0; mu = 2;
        disc = 1; G = 0;
        for k = 1:horizon
            s = state_id(x,B,m,mu,Bmax,mmax);
            probs = softmax(theta(s,1:numActs(s))/temp);
            ai = sample_discrete(probs);
            a = actMat(s,ai);
            Bp = B - a*x;
            mp = m + a;
            muNext = sample_discrete(P_M(mu,:));
            omega = omega_vals(sample_discrete(P_noise(mu,:)));
            xp = price_update(x, omega, x_bar);
            r = mp*(xp-x);
            sp = state_id(xp,Bp,mp,muNext,Bmax,mmax);
            delta = r + gamma*Vcritic(sp) - Vcritic(s);
            Vcritic(s) = Vcritic(s) + vLR*delta;
            deltaActor = max(min(delta, clipDelta), -clipDelta);
            for j = 1:numActs(s)
                grad = -probs(j);
                if j == ai
                    grad = grad + 1;
                end
                theta(s,j) = theta(s,j) + thetaLR*deltaActor*grad;
            end
            visits(s,ai) = visits(s,ai) + 1;
            G = G + disc*r;
            disc = disc*gamma;
            x=xp; B=Bp; m=mp; mu=muNext;
            if x == 0
                break;
            end
        end
        trainReturns(ep) = G;
    end
    policy = holdIdx;
    for s = 1:N
        if sum(visits(s,1:numActs(s))) > 0
            [~, policy(s)] = max(theta(s,1:numActs(s)));
        end
    end
end

function p = softmax(z)
    z = z - max(z);
    ez = exp(z);
    p = ez / sum(ez);
end
