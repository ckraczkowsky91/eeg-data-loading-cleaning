import os

!pip install mne
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)

# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=120.0).pick(picks=["mag", "eeg", "stim", "eog"])
raw.load_data()
print('RAW DATA', raw.info)

mag_channels = mne.pick_types(raw.info, meg='mag')
eeg_channels = mne.pick_types(raw.info, eeg=True)

### Observe the raw data

## Detect low-drift frequencies using amplitude plot
raw.plot(picks=eeg_channels, n_channels=len(eeg_channels),
         start=0, duration=120, scalings={'eeg': 50e-6})

## Detect power line noise using PSD plot
raw.compute_psd(fmax=75).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)


## Observe potential EOG artifacts (e.g. eye blink) or ECG artifact (e.g. heartbeat)
regexp = r"(MEG [12][45][123]1|EEG 00.)"
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)

## Observe EOG artifacts
eog_evoked = create_eog_epochs(raw).average()             #Use one of the functions imported from mne to create epochs which automatically detects ocular events and creates epochs around them, then compute an average over the epochs to show an example of an eye blink in the data, returns Evoked
eog_evoked.apply_baseline(baseline=(None, -0.2))          #Want to apply baseline to smooth peaks. Epochs are -0.5 and 0.5 seconds from the identified ocular movement, grab the baseline just before the eye movement starts (-0.2 before event) to not include the event
eog_evoked.plot_joint()                                   #Plot evoked data as butterfly plot and add topomaps for time points, we should see spikes in the butterfly plot that correspond to activity around the frontal lobe

## Observe ECG artifacts
ecg_evoked = create_ecg_epochs(raw).average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))
ecg_evoked.plot_joint()

### Pre-filter data (apply band-pass + notch filtering

## Apply filter
filtered_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

## Validate low-frequency drift smoothing
filtered_raw.plot(picks=eeg_channels, n_channels=len(eeg_channels),
         start=0, duration=120, scalings={'eeg': 50e-6})


### Run ICA to create an ICA object
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filtered_raw)

## Identify artifacts from components
ica.plot_sources(filtered_raw, show_scrollbars=False)
ica.plot_components()

ica.plot_overlay(raw, exclude=[0], picks='eeg') # Look at the raw original data against the cleaned dat (ICA000 removed) applied to just the EEG channels
ica.plot_overlay(raw, exclude=[1], picks='mag')

ica.plot_overlay(raw, exclude=[4], picks='eeg')
ica.plot_overlay(raw, exclude=[4], picks='mag')

### Clean the raw data
ica.exclude = [0, 1]

raw_copy = raw.copy()
ica.apply(raw_copy)                             # apply the ICA analysis to the raw data

## Validate cleaning
raw.plot(order=artifact_picks, n_channels=len(artifact_picks))                                      # see the plot of the original data looking at the subset of channels that we looked at at the beginning
raw_copy.plot(order=artifact_picks, n_channels=len(artifact_picks))                                 # see the plot of the cleaned data
