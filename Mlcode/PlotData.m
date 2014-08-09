function [ ] = PlotData (X,Y)

gscatter(X(:,2), X(:,3), Y,'br','xo');
xlabel('Feature 1');
ylabel('Feature 2');

fprintf( '\n Press Enter to Continue \n ');
pause;