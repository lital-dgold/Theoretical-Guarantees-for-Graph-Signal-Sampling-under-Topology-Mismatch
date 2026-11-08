function [sampling_vector] = samplingSet2vector(sampling_set, n)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    sampling_vector = zeros(1,n);
    sampling_vector(sampling_set) = 1;
    sampling_vector = logical(sampling_vector);
end

