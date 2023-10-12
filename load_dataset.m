function [X, y] = load_dataset(path)
  X_struct = load (path, "X");
  y_struct = load (path, "y");
  X = X_struct.X;
  y = y_struct.y;
endfunction
