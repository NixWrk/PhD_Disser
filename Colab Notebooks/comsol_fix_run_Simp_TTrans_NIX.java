import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/**
 * COMSOL 6.2 runner for Simp_TTrans_NIX.mph.
 *
 * Use a working copy in C:\tmp to avoid COMSOL Java security restrictions on
 * arbitrary file-system reads. The original source file is:
 *   Z:\02 Big_data\Comsol\Template\Simp_TTrans_NIX.mph
 *
 * What it fixes:
 * 1. Keeps the heart sphere center fixed while sweeping radius.
 *    Old model: Rh = 52.5[mm]-delRh and yh = 78[mm]-Rh+delY,
 *    so delRh changed both radius and center.
 * 2. Uses Nik solution-cloud resistivities:
 *    rho1 -> soft tissue -> roMuscle,
 *    rho2 -> lung        -> roLung.
 * 3. Uses TRKG channel-1 targets:
 *    Z1 = 12.703 ohm at systole start,
 *    Z2 = 12.626 ohm at systole end.
 * 4. Removes shared domain 12 from the two current terminals.
 *
 * Run with COMSOL 6.2, not 6.0:
 *   comsolbatch -inputfile comsol_fix_run_Simp_TTrans_NIX.java
 */
public class comsol_fix_run_Simp_TTrans_NIX {
  private static final String INPUT_MPH =
      "C:\\tmp\\Simp_TTrans_NIX.mph";
  private static final String OUTPUT_MPH =
      "C:\\tmp\\Simp_TTrans_NIX_fixed_6_2.mph";
  private static final String R0_TABLE =
      "C:\\tmp\\Simp_TTrans_NIX_R0_calibration.csv";
  private static final String DELRH_TABLE =
      "C:\\tmp\\Simp_TTrans_NIX_delRh_sweep.csv";

  // After inspecting R0_TABLE, set this to the Rh0 that best matches Z1_TARGET.
  private static final String R0_FOR_DELRH_SWEEP = "52.5[mm]";

  public static void main(String[] args) throws Exception {
    ModelUtil.showProgress(true);
    Model model = ModelUtil.load("SimpTTransNIX", INPUT_MPH);

    applyFixes(model);

    // Stage 1: estimate baseline radius R0 from Z1 with delRh = 0.
    model.param().set("delRh", "0[mm]");
    runParamSweep(
        model,
        "Rh0",
        "range(40,1,65)",
        "mm",
        R0_TABLE);

    // Stage 2: preliminary radius-change sweep at the chosen R0.
    // Replace R0_FOR_DELRH_SWEEP after reading the R0 calibration table.
    model.param().set("Rh0", R0_FOR_DELRH_SWEEP);
    runParamSweep(
        model,
        "delRh",
        "range(-5,0.25,10)",
        "mm",
        DELRH_TABLE);

    model.save(OUTPUT_MPH);
  }

  private static void applyFixes(Model model) {
    // Experimental TRKG channel-1 targets for 90nik.csv, breath-hold inhale.
    model.param().set("Z1_target", "12.703[ohm]");
    model.param().set("Z2_target", "12.626[ohm]");
    model.param().set("dZ_target", "Z2_target-Z1_target");

    // Nik solution-cloud resistivities: rho1 is soft tissue, rho2 is lung.
    model.param().set("rho1_soft", "6.84[ohm*m]");
    model.param().set("rho2_lung", "10.36[ohm*m]");
    model.param().set("roMuscle", "rho1_soft");
    model.param().set("roLung", "rho2_lung");
    model.param().set("I", "1[A]");

    // Fixed-center heart sphere parameterization.
    model.param().set("Rh0", "52.5[mm]");
    model.param().set("yh0", "78[mm]-Rh0");
    model.param().set("Rh", "Rh0-delRh");
    model.param().set("Rb", "Rh-10[mm]");
    model.param().set("yh", "yh0+delY");

    // The current file has domain 12 in both current terminals. That makes the
    // injection path ambiguous. Keep only the likely current electrode domains.
    model.component("comp1").physics("ec").feature("term1").selection().set(new int[] {11});
    model.component("comp1").physics("ec").feature("term2").selection().set(new int[] {1});

    // Make sure the saved Z expression is the one we compare with TRKG.
    model.component("comp1").variable("var1").set("ZTT", "abs(ElR-ElL)/I");
  }

  private static void runParamSweep(
      Model model,
      String parameter,
      String values,
      String unit,
      String outputCsv) {
    model.study("std2").feature("param1").active(true);
    model.study("std2").feature("param1").set("sweeptype", "filled");
    model.study("std2").feature("param1").set("pname", new String[] {parameter});
    model.study("std2").feature("param1").set("plistarr", new String[] {values});
    model.study("std2").feature("param1").set("punit", new String[] {unit});

    model.study("std2").run();

    model.result().numerical("gev1").set("expr", new String[] {
        "ZTT",
        "ZTT-Z1_target",
        "ZTT-Z2_target",
        "abs(ZTT-Z1_target)",
        "abs(ZTT-Z2_target)"
    });
    model.result().numerical("gev1").set("unit", new String[] {
        "ohm",
        "ohm",
        "ohm",
        "ohm",
        "ohm"
    });
    model.result().numerical("gev1").set("descr", new String[] {
        "ZTT",
        "ZTT minus TRKG systole-start target",
        "ZTT minus TRKG systole-end target",
        "absolute start-target error",
        "absolute end-target error"
    });
    model.result().numerical("gev1").set("table", "tbl3");
    model.result().table("tbl3").clearTableData();
    model.result().numerical("gev1").setResult();
    model.result().table("tbl3").save(outputCsv);
  }
}
