function []=imwrite_single(data,name)

outputFileName = name;

t = Tiff(outputFileName,'w');

tagstruct.ImageLength     = size(data,1);
tagstruct.ImageWidth      = size(data,2);
tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample   = 32;
tagstruct.SampleFormat = 3;
tagstruct.SamplesPerPixel = size(data,3);
tagstruct.RowsPerStrip    = 16;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Software        = 'MATLAB';
tagstruct.Compression = Tiff.Compression.LZW;
t.setTag(tagstruct)


t.write(data);
t.close();
