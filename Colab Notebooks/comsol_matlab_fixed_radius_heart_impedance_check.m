%% Fixed-radius heart impedance check via COMSOL LiveLink for MATLAB
% Goal:
%   Keep the CT/COMSOL geometry fixed, keep heart sphere radius fixed,
%   use CT lung rho, and explain Z1/Z2 only by changing effective
%   heart-sphere resistivity.
%
% How to run:
%   1. Start COMSOL Multiphysics server, or let mphstart connect to one.
%   2. Open this script in MATLAB with COMSOL LiveLink installed.
%   3. Set MODEL_PATH to the GUI-cleaned MPH model.
%
% Expected model conventions:
%   - Global variable/expression ZTT exists and evaluates trans-thoracic Z.
%   - Parameters rho1_soft, rho2_lung, rho_myocardium, rho_blood exist.
%   - Parameters Rh0 and delRh exist if the model still has radius controls.
%   - Materials use the parameters above.

clear; clc;

MODEL_PATH = 'C:\tmp\TTrans_NIX_point_radius_6_0.mph';
OUT_CSV = 'C:\tmp\TTrans_NIX_matlab_fixed_radius_heart_impedance.csv';

COMSOL_MLI = ['C:\PC\COMSOL.Multiphysics.6.0.318.Win.Linux.macOS-SSQ\', ...
    'Installed_comsol\COMSOL60\Multiphysics\mli'];
% Resolve the repository root from this script's own location, so the path
% stays valid on any machine and after the checkout has been moved.
REPO_ROOT = fileparts(mfilename('fullpath'));
while exist(fullfile(REPO_ROOT, '.git'), 'dir') ~= 7
    PARENT_DIR = fileparts(REPO_ROOT);
    if strcmp(PARENT_DIR, REPO_ROOT)
        error('comsol_check:noRepoRoot', ...
            'Repository root (.git) not found above %s', mfilename('fullpath'));
    end
    REPO_ROOT = PARENT_DIR;
end
PARAM_DIR = fullfile(REPO_ROOT, 'Colab Notebooks', 'params');

SUBJECT = 'РќРёРє';   % static.json/ct.json are currently mojibake-encoded in this checkout.

Z1_TARGET = 12.703;   % Ohm, TRKG at systole start
Z2_TARGET = 12.626;   % Ohm, TRKG at systole end

addpath(COMSOL_MLI);
import com.comsol.model.*
import com.comsol.model.util.*

try
    mphstart;
catch ME
    fprintf('mphstart warning: %s\n', ME.message);
end

ct = jsondecode(fileread(fullfile(PARAM_DIR, 'ct.json')));
st = jsondecode(fileread(fullfile(PARAM_DIR, 'static.json')));

rho_lung_ct = ct.(SUBJECT).rho2_ct;
rho_soft_static = st.(SUBJECT).rho1;

model = mphload(MODEL_PATH);

% Fixed geometry: no radius change. If these parameters do not exist in the
% GUI model, COMSOL will throw; in that case comment this block out.
try
    model.param.set('Rh0', '42.5[mm]');
    model.param.set('delRh', '0[mm]');
    model.param.set('heartScale', '1');
catch
    fprintf('Radius parameters not found; assuming GUI geometry is already fixed.\n');
end

% CT/literature material point for this check.
model.param.set('rho1_soft', sprintf('%.12g[ohm*m]', rho_soft_static));
model.param.set('rho2_lung', sprintf('%.12g[ohm*m]', rho_lung_ct));

% Sweep effective heart-sphere rho. We set myocardium and blood together on
% purpose: this test asks whether a fixed-size sphere with changed impedance
% can account for Z1/Z2, not whether blood/myocardium separately can.
rho_heart_grid = logspace(log10(0.05), log10(20), 80);
Z = nan(size(rho_heart_grid));

for k = 1:numel(rho_heart_grid)
    rho_h = rho_heart_grid(k);
    model.param.set('rho_myocardium', sprintf('%.12g[ohm*m]', rho_h));
    model.param.set('rho_blood', sprintf('%.12g[ohm*m]', rho_h));
    try
        model.study('std1').run;
        Z(k) = mphglobal(model, 'ZTT', 'unit', 'ohm');
        fprintf('%3d/%3d rho_heart=%.6g ohm*m -> Z=%.9g ohm\n', ...
            k, numel(rho_heart_grid), rho_h, Z(k));
    catch ME
        fprintf('%3d/%3d rho_heart=%.6g failed: %s\n', ...
            k, numel(rho_heart_grid), rho_h, ME.message);
    end
end

valid = isfinite(Z);
rho_valid = rho_heart_grid(valid);
Z_valid = Z(valid);

% Sort by Z for interpolation; if the curve is almost flat/nonmonotonic, the
% interpolation result is diagnostic rather than final.
[Zs, order] = sort(Z_valid);
rhos = rho_valid(order);

rho_at_Z1 = interp1(Zs, rhos, Z1_TARGET, 'linear', 'extrap');
rho_at_Z2 = interp1(Zs, rhos, Z2_TARGET, 'linear', 'extrap');
delta_rho = rho_at_Z2 - rho_at_Z1;

T = table(rho_heart_grid(:), Z(:), ...
    'VariableNames', {'rho_heart_ohm_m', 'ZTT_ohm'});
writetable(T, OUT_CSV);

fprintf('\nTargets:\n');
fprintf('  Z1 = %.6f ohm -> rho_heart = %.9g ohm*m\n', Z1_TARGET, rho_at_Z1);
fprintf('  Z2 = %.6f ohm -> rho_heart = %.9g ohm*m\n', Z2_TARGET, rho_at_Z2);
fprintf('  delta_rho_heart = %.9g ohm*m\n', delta_rho);
fprintf('Saved sweep: %s\n', OUT_CSV);
