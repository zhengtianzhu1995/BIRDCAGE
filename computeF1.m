function [F1] = computeF1(beta, betaT)
supp = find(betaT);
esupp = find(beta);
fp = length(setdiff(esupp,supp));
fn = length(setdiff(supp,esupp));
tp = length(intersect(esupp, supp));
F1 = 2*tp/(2*tp + fp + fn);
end