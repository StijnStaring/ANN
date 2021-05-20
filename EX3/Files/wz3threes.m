clear;
load threes -ascii;
data = threes;
colormap('gray')

covmatrix = cov(data);

for q = 100
    
   
    [eigenvectors, eigenvalues] =eigs(covmatrix, q);
    eigenvalues = diag(eigenvalues);
    projectionmatrix = eigenvectors';

    data2 = data';
    reduceddata = projectionmatrix * data2;
    recreateddata = eigenvectors * reduceddata;

    recreateddata =recreateddata';

    imagesc(reshape(recreateddata(3,:),16,16),[0,1])

    error = sqrt(mean(mean((recreateddata - data).^2)))

end







