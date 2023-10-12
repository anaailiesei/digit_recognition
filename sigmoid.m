% Functie vectorizata pentru calcularea sigmoidului.

function [y] = sigmoid(x)
  y = 1 ./ (1 + e .^ (-x));
endfunction
