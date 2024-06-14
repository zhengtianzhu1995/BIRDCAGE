function error = computeError(beta, betaT, varargin)
error = norm(beta-betaT,varargin{:});
end