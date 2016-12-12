% compute the first derivative of theta(1) beta density (without the denominator Beta(theta(1), theta(2)) for a vector

function deriOfFirstPar_BetaPdfForVec = deriOfFirstPar_BetaPdfForVec(x,theta)

m = length(x);

for i = 1:m
    x(i) = (x(i))^(exp(theta(1))-1)*(1-x(i))^(exp(theta(2))-1) * ...
        log(x(i)) * exp(theta(1));
end


deriOfFirstPar_BetaPdfForVec = x;


end