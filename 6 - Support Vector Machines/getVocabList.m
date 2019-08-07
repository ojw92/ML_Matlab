function vocabList = getVocabList()
%GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
%cell array of the words
%   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
%   and returns a cell array of the words in vocabList.


%% Read the fixed vocabulary list
% open the file for read access
fid = fopen('vocab.txt');

% Store all dictionary words in cell array vocab{}
n = 1899;  % Total number of words in the dictionary

% For ease of implementation, use a struct to map the strings => integers
% In practice, should use some form of hashmap
% Create a cell array
vocabList = cell(n, 1);
for i = 1:n
    % Word Index (can ignore since it will be = i)
    % Read the data from text file 'fid', reading 1 element according to format %d 
    fscanf(fid, '%d', 1);
    % Actual Word
    % Reading the file for a string (characters)
    vocabList{i} = fscanf(fid, '%s', 1);
end
fclose(fid);

end
