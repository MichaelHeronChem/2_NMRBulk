Features to implement: 



Current pipeline:

Takes in a raw NMR file and applies the following processing::::::::
Calculates PPM axis using spectrometer frequency and sweep width
Does not apply line broadening (not needed)
Applies zero filling
Applies fourier transform
Phases using an auto edge-symmetry algorithm:
-- Anchors to the aldehyde peak, adjusts p0 value of the phase until the bottoms of the peak are level, then adjusts p1 to maximise the horizontalness of the peaks above 5ppm (the supressed region is masked)
Then does autobaseline correction on unmasked region
Uses unsupressed scoutfid to calculate reference shift using 1.96 ppm as the shift for the undeuterated acetonitrile singlet signal
Applies shift to actual spectrum

Saves as a png