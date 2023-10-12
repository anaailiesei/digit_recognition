function [matrix] = initialize_weights(L_prev, L_next)
  % Se calculeaza epsilon
  epsilon = sqrt(6) / sqrt(L_prev + L_next);

  % Se creeaza matricea cu valori random
  matrix = rand(L_next, L_prev + 1);
  matrix *= 2;
  matrix -= 1;
  matrix *= epsilon;
endfunction
