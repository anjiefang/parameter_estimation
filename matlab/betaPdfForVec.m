% compute the beta density (without the denominator Beta(theta(1), theta(2)) for a vector

function betaPdfForVec = betaPdfForVec(x,theta)

m = length(x);

for i = 1:m
    x(i) = (x(i))^(exp(theta(1))-1)*(1-x(i))^(exp(theta(2))-1);
end


betaPdfForVec = x;


end