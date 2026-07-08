function centroids = element_centroids(fmdl)
%ELEMENT_CENTROIDS Return centroids of tetrahedral elements.

nodes = fmdl.nodes;
elems = fmdl.elems;
centroids = (nodes(elems(:,1),:) + nodes(elems(:,2),:) + ...
             nodes(elems(:,3),:) + nodes(elems(:,4),:)) / 4;
end
