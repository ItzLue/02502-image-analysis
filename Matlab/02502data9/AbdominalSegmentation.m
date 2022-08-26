function [ISpleen, ILiver, IKidney] = AbdominalSegmentation(ct)

T1 = 80;
T2 = 150;

binI = (ct > T1) & (ct < T2);

se = strel('disk',2);
Iclose = imclose(binI, se);

se = strel('disk',3);
Iopen = imopen(Iclose, se);

L8 = bwlabel(Iopen,8);
RGB8 = label2rgb(L8);

stats8 = regionprops(L8, 'All');

allArea = [stats8.Area];
allPerimeter = [stats8.Perimeter];
allEcc = [stats8.Eccentricity];
allCompactness = [stats8.Perimeter] ./ [stats8.Area];

idx = find([stats8.Area] > 1000 & [stats8.Perimeter] < 400);

ISpleen = ismember(L8,idx);
ILiver = ismember(L8,idx);
IKidney = ismember(L8,idx);

