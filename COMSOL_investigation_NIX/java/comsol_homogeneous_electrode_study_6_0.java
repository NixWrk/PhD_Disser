import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class comsol_homogeneous_electrode_study_6_0 {
  private static final String STEP_PATH = "C:\\tmp\\TTrans_for_NIX.step";
  private static final String OUT_DIR = "C:\\tmp\\TTrans_NIX_homogeneous_electrode_study";
  private static final String SUMMARY_CSV = OUT_DIR + "\\homogeneous_electrode_summary.csv";

  private static final double HEART_CX = -0.026;
  private static final double HEART_CY = 0.0355;
  private static final double HEART_CZ = 0.026;
  private static final double BODY_TOP_Y = -0.026;
  private static final double BODY_TOP_Z = 0.184;
  private static final double ARM_SKIN_Y = -0.050;
  private static final double ARM_SKIN_Z = 0.208;

  private static void ensureOutDir() {
    File dir = new File(OUT_DIR);
    if (!dir.exists()) {
      dir.mkdirs();
    }
  }

  private static void material(Model model, String tag, String label, String sigmaExpr, int[] domains) {
    model.component("comp1").material().create(tag, "Common");
    model.component("comp1").material(tag).label(label);
    model.component("comp1").material(tag).selection().set(domains);
    model.component("comp1").material(tag).propertyGroup("def").set("electricconductivity", new String[]{sigmaExpr});
    model.component("comp1").material(tag).propertyGroup("def").set("relpermittivity", new String[]{"1"});
  }

  private static double[][] measureDomains(Model model) {
    int[] nEntities = model.component("comp1").geom("geom1").getNEntities();
    int domainCount = nEntities.length > 3 ? nEntities[3] : 0;
    double[][] domains = new double[domainCount][8];
    for (int domain = 1; domain <= domainCount; domain++) {
      model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 3);
      model.component("comp1").geom("geom1").measureFinal().selection().set(domain);
      double[] bbox = model.component("comp1").geom("geom1").measureFinal().getBoundingBox();
      domains[domain - 1][0] = domain;
      domains[domain - 1][1] = model.component("comp1").geom("geom1").measureFinal().getVolume();
      domains[domain - 1][2] = bbox[0];
      domains[domain - 1][3] = bbox[1];
      domains[domain - 1][4] = bbox[2];
      domains[domain - 1][5] = bbox[3];
      domains[domain - 1][6] = bbox[4];
      domains[domain - 1][7] = bbox[5];
    }
    return domains;
  }

  private static int domainId(double[] d) {
    return (int)Math.round(d[0]);
  }

  private static double volume(double[] d) {
    return d[1];
  }

  private static double cx(double[] d) {
    return 0.5 * (d[2] + d[3]);
  }

  private static double cy(double[] d) {
    return 0.5 * (d[4] + d[5]);
  }

  private static double cz(double[] d) {
    return 0.5 * (d[6] + d[7]);
  }

  private static double dx(double[] d) {
    return d[3] - d[2];
  }

  private static double pcx(double[] bbox) {
    return 0.5 * (bbox[0] + bbox[1]);
  }

  private static double pcy(double[] bbox) {
    return 0.5 * (bbox[2] + bbox[3]);
  }

  private static double pcz(double[] bbox) {
    return 0.5 * (bbox[4] + bbox[5]);
  }

  private static double bdx(double[] bbox) {
    return bbox[1] - bbox[0];
  }

  private static double center(double lo, double hi) {
    return 0.5 * (lo + hi);
  }

  private static double intervalDistance(double value, double lo, double hi) {
    if (value < lo) {
      return lo - value;
    }
    if (value > hi) {
      return value - hi;
    }
    return 0.0;
  }

  private static int closestElectrodeDomain(double[][] domains, double targetX) {
    int best = -1;
    double bestDist = Double.POSITIVE_INFINITY;
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      double dist = Math.abs(cx(d) - targetX);
      if (dx(d) < 0.03 && volume(d) < 5.0e-4 && dist < bestDist) {
        best = domainId(d);
        bestDist = dist;
      }
    }
    if (best < 0 || bestDist > 0.06) {
      throw new RuntimeException("Could not classify old electrode domain near x=" + targetX);
    }
    return best;
  }

  private static int closestPadDomain(double[][] domains, double targetX, double targetY, double targetZ) {
    int best = -1;
    double bestDist = Double.POSITIVE_INFINITY;
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      double x = cx(d) - targetX;
      double y = cy(d) - targetY;
      double z = cz(d) - targetZ;
      double dist = Math.sqrt(x * x + y * y + z * z);
      if (dx(d) < 0.025 && volume(d) > 1.0e-8 && volume(d) < 5.0e-6 && dist < bestDist) {
        best = domainId(d);
        bestDist = dist;
      }
    }
    if (best < 0 || bestDist > 0.025) {
      throw new RuntimeException("Could not classify pad domain near x=" + targetX + ", bestDist=" + bestDist);
    }
    return best;
  }

  private static int closestPoint(Model model, double targetX) {
    int[] nEntities = model.component("comp1").geom("geom1").getNEntities();
    int pointCount = nEntities.length > 0 ? nEntities[0] : 0;
    int best = -1;
    double bestDist = Double.POSITIVE_INFINITY;
    for (int point = 1; point <= pointCount; point++) {
      model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 0);
      model.component("comp1").geom("geom1").measureFinal().selection().set(point);
      double[] bbox = model.component("comp1").geom("geom1").measureFinal().getBoundingBox();
      double x = pcx(bbox) - targetX;
      double y = pcy(bbox) - ARM_SKIN_Y;
      double z = pcz(bbox) - ARM_SKIN_Z;
      double dist = Math.sqrt(x * x + y * y + z * z);
      if (dist < bestDist) {
        best = point;
        bestDist = dist;
      }
    }
    if (best < 0 || bestDist > 1.0e-6) {
      throw new RuntimeException("Could not classify point electrode near x=" + targetX + ", bestDist=" + bestDist);
    }
    return best;
  }

  private static int largestDomain(double[][] domains) {
    int best = -1;
    double bestVolume = -1.0;
    for (int i = 0; i < domains.length; i++) {
      if (volume(domains[i]) > bestVolume) {
        best = domainId(domains[i]);
        bestVolume = volume(domains[i]);
      }
    }
    return best;
  }

  private static int bodyTopFace(Model model, int softDomain, double targetX) {
    int[] boundaries = model.component("comp1").geom("geom1").getAdj(3, 2, softDomain);
    int best = -1;
    double bestScore = Double.POSITIVE_INFINITY;
    double[] bestBbox = null;
    for (int i = 0; i < boundaries.length; i++) {
      int b = boundaries[i];
      model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 2);
      model.component("comp1").geom("geom1").measureFinal().selection().set(b);
      double[] bbox = model.component("comp1").geom("geom1").measureFinal().getBoundingBox();
      double xPenalty = intervalDistance(targetX, bbox[0], bbox[1]);
      double yPenalty = Math.abs(center(bbox[2], bbox[3]) - BODY_TOP_Y);
      double zPenalty = Math.abs(center(bbox[4], bbox[5]) - BODY_TOP_Z);
      double score = 100.0 * xPenalty + 20.0 * bdx(bbox) + yPenalty + zPenalty;
      if (score < bestScore) {
        best = b;
        bestScore = score;
        bestBbox = bbox;
      }
    }
    if (best < 0) {
      throw new RuntimeException("Could not find body top face near x=" + targetX);
    }
    System.out.println("BODY_TOP_FACE,targetX=" + targetX + ",boundary=" + best + ",bbox=" + Arrays.toString(bestBbox));
    return best;
  }

  private static boolean contains(int[] values, int id) {
    for (int i = 0; i < values.length; i++) {
      if (values[i] == id) {
        return true;
      }
    }
    return false;
  }

  private static int[] allExcept(int domainCount, int[] excluded) {
    List<Integer> out = new ArrayList<Integer>();
    for (int id = 1; id <= domainCount; id++) {
      if (!contains(excluded, id)) {
        out.add(Integer.valueOf(id));
      }
    }
    int[] result = new int[out.size()];
    for (int i = 0; i < out.size(); i++) {
      result[i] = out.get(i).intValue();
    }
    return result;
  }

  private static void createPad(Model model, String tag, double x, double y, double z, String radius) {
    model.component("comp1").geom("geom1").create(tag, "Sphere");
    model.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{x, y, z});
    model.component("comp1").geom("geom1").feature(tag).set("r", radius);
  }

  private static void createPoint(Model model, String tag, double x) {
    model.component("comp1").geom("geom1").create(tag, "Point");
    model.component("comp1").geom("geom1").feature(tag).set("p", new double[]{x, ARM_SKIN_Y, ARM_SKIN_Z});
  }

  private static Model baseModel(String scenario, boolean pointElectrodes, String padMode) {
    Model model = ModelUtil.create("hom_" + scenario);
    model.modelPath(OUT_DIR);
    model.label("hom_" + scenario + ".mph");

    model.param().set("I", "1[A]");
    model.param().set("rho_soft", "4.728[ohm*m]");
    model.param().set("sigma_electrode", "4.032e6[S/m]");
    model.param().set("Rh_geom", "42.5[mm]");
    model.param().set("Rh0", "42.5[mm]");
    model.param().set("delRh", "0[mm]");
    model.param().set("heartRadius", "Rh0-delRh");
    model.param().set("heartScale", "(Rh0-delRh)/Rh_geom");
    model.param().set("fillOuterRadius", "max(Rh_geom,heartRadius)");

    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").geom("geom1").create("imp1", "Import");
    model.component("comp1").geom("geom1").feature("imp1").set("filename", STEP_PATH);
    model.component("comp1").geom("geom1").feature("imp1").set("unit", "source");
    model.component("comp1").geom("geom1").feature("imp1").importData();

    model.component("comp1").geom("geom1").create("scaHeart", "Scale");
    model.component("comp1").geom("geom1").feature("scaHeart").selection("input").set("imp1.id14", "imp1.id17");
    model.component("comp1").geom("geom1").feature("scaHeart").set("factor", "heartScale");
    model.component("comp1").geom("geom1").feature("scaHeart").set("pos", new double[]{HEART_CX, HEART_CY, HEART_CZ});

    model.component("comp1").geom("geom1").create("sphFillOuter", "Sphere");
    model.component("comp1").geom("geom1").feature("sphFillOuter").set("pos", new double[]{HEART_CX, HEART_CY, HEART_CZ});
    model.component("comp1").geom("geom1").feature("sphFillOuter").set("r", "fillOuterRadius");
    model.component("comp1").geom("geom1").create("sphFillInner", "Sphere");
    model.component("comp1").geom("geom1").feature("sphFillInner").set("pos", new double[]{HEART_CX, HEART_CY, HEART_CZ});
    model.component("comp1").geom("geom1").feature("sphFillInner").set("r", "heartRadius");
    model.component("comp1").geom("geom1").create("difSoftFill", "Difference");
    model.component("comp1").geom("geom1").feature("difSoftFill").selection("input").set("sphFillOuter");
    model.component("comp1").geom("geom1").feature("difSoftFill").selection("input2").set("sphFillInner");

    if (pointElectrodes) {
      createPoint(model, "ptA_ground", -0.295);
      createPoint(model, "ptM_left", -0.197);
      createPoint(model, "ptN_right", 0.197);
      createPoint(model, "ptB_source", 0.295);
    }
    if ("body_top".equals(padMode)) {
      createPad(model, "padA_ground", -0.295, BODY_TOP_Y, BODY_TOP_Z, "3[mm]");
      createPad(model, "padM_left", -0.197, BODY_TOP_Y, BODY_TOP_Z, "3[mm]");
      createPad(model, "padN_right", 0.197, BODY_TOP_Y, BODY_TOP_Z, "3[mm]");
      createPad(model, "padB_source", 0.295, BODY_TOP_Y, BODY_TOP_Z, "3[mm]");
    } else if ("arm_top".equals(padMode)) {
      createPad(model, "padA_ground", -0.295, ARM_SKIN_Y, ARM_SKIN_Z, "4[mm]");
      createPad(model, "padM_left", -0.197, ARM_SKIN_Y, ARM_SKIN_Z, "4[mm]");
      createPad(model, "padN_right", 0.197, ARM_SKIN_Y, ARM_SKIN_Z, "4[mm]");
      createPad(model, "padB_source", 0.295, ARM_SKIN_Y, ARM_SKIN_Z, "4[mm]");
    }

    model.component("comp1").geom("geom1").run();
    model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
    return model;
  }

  private static void assignHomogeneousMaterials(Model model, int[] electrodeDomains) {
    int domainCount = model.component("comp1").geom("geom1").getNEntities()[3];
    int[] softDomains = allExcept(domainCount, electrodeDomains);
    material(model, "matSoft", "homogeneous soft rho", "1/rho_soft", softDomains);
    material(model, "matElectrode", "electrode metal", "sigma_electrode", electrodeDomains);
    System.out.println("MATERIALS,soft=" + Arrays.toString(softDomains) + ",electrodes=" + Arrays.toString(electrodeDomains));
  }

  private static void addAverageCouplings(Model model, int geomDim, int[] leftSelection, int[] rightSelection) {
    model.component("comp1").cpl().create("aveVL", "Average");
    model.component("comp1").cpl("aveVL").selection().geom("geom1", geomDim);
    model.component("comp1").cpl("aveVL").selection().set(leftSelection);
    model.component("comp1").cpl().create("aveVR", "Average");
    model.component("comp1").cpl("aveVR").selection().geom("geom1", geomDim);
    model.component("comp1").cpl("aveVR").selection().set(rightSelection);
  }

  private static void finishSolveSetup(Model model) {
    model.component("comp1").variable().create("var1");
    model.component("comp1").variable("var1").set("VL", "aveVL(V)");
    model.component("comp1").variable("var1").set("VR", "aveVR(V)");
    model.component("comp1").variable("var1").set("ZTT", "abs(VR-VL)/I");

    model.component("comp1").mesh("mesh1").autoMeshSize(5);
    model.component("comp1").mesh("mesh1").run();

    model.study().create("std1");
    model.study("std1").create("stat", "Stationary");
    model.result().numerical().create("gevZ", "EvalGlobal");
    model.result().numerical("gevZ").set("expr", new String[]{"ZTT", "VR", "VL"});
    model.result().numerical("gevZ").set("unit", new String[]{"ohm", "V", "V"});
  }

  private static double[] solve(Model model) {
    model.study("std1").run();
    model.result().numerical("gevZ").set("data", "dset1");
    double[][] real = model.result().numerical("gevZ").getReal();
    if (real.length < 3 || real[0].length < 1) {
      throw new RuntimeException("EvalGlobal returned an empty result");
    }
    return new double[]{real[0][0], real[1][0], real[2][0]};
  }

  private static void exportGeometry(Model model, String scenario) {
    String path = OUT_DIR + "\\" + scenario + "_geometry.png";
    model.component("comp1").geom("geom1").image().set("pngfilename", path);
    model.component("comp1").geom("geom1").image().set("width", "1600");
    model.component("comp1").geom("geom1").image().set("height", "1000");
    model.component("comp1").geom("geom1").image().set("unit", "px");
    model.component("comp1").geom("geom1").image().export();
    System.out.println("EXPORTED=" + path);
  }

  private static void exportMaterial(Model model, String scenario, String tag) {
    String path = OUT_DIR + "\\" + scenario + "_" + tag + ".png";
    model.component("comp1").material(tag).image().set("pngfilename", path);
    model.component("comp1").material(tag).image().set("width", "1600");
    model.component("comp1").material(tag).image().set("height", "1000");
    model.component("comp1").material(tag).image().set("unit", "px");
    model.component("comp1").material(tag).image().export();
    System.out.println("EXPORTED=" + path);
  }

  private static void exportPhysicsFeature(Model model, String scenario, String tag) {
    String path = OUT_DIR + "\\" + scenario + "_phys_" + tag + ".png";
    model.component("comp1").physics("ec").feature(tag).image().set("pngfilename", path);
    model.component("comp1").physics("ec").feature(tag).image().set("width", "1600");
    model.component("comp1").physics("ec").feature(tag).image().set("height", "1000");
    model.component("comp1").physics("ec").feature(tag).image().set("unit", "px");
    model.component("comp1").physics("ec").feature(tag).image().export();
    System.out.println("EXPORTED=" + path);
  }

  private static void exportCoupling(Model model, String scenario, String tag) {
    String path = OUT_DIR + "\\" + scenario + "_cpl_" + tag + ".png";
    model.component("comp1").cpl(tag).image().set("pngfilename", path);
    model.component("comp1").cpl(tag).image().set("width", "1600");
    model.component("comp1").cpl(tag).image().set("height", "1000");
    model.component("comp1").cpl(tag).image().set("unit", "px");
    model.component("comp1").cpl(tag).image().export();
    System.out.println("EXPORTED=" + path);
  }

  private static void exportCommonImages(Model model, String scenario, String[] physicsFeatures) {
    exportGeometry(model, scenario);
    exportMaterial(model, scenario, "matSoft");
    exportMaterial(model, scenario, "matElectrode");
    for (int i = 0; i < physicsFeatures.length; i++) {
      exportPhysicsFeature(model, scenario, physicsFeatures[i]);
    }
    exportCoupling(model, scenario, "aveVL");
    exportCoupling(model, scenario, "aveVR");
  }

  private static void saveAndRecord(PrintWriter out, Model model, String scenario, double[] result, String notes) throws IOException {
    model.save(OUT_DIR + "\\" + scenario + ".mph");
    out.println(scenario + "," + result[0] + "," + result[1] + "," + result[2] + ",\"" + notes + "\"");
    out.flush();
    System.out.println("RESULT," + scenario + "," + Arrays.toString(result));
  }

  private static void runOldDomain(PrintWriter out) throws IOException {
    String scenario = "old_domain_electrodes";
    Model model = baseModel(scenario, false, "");
    double[][] domains = measureDomains(model);
    int ground = closestElectrodeDomain(domains, -0.295);
    int voltLeft = closestElectrodeDomain(domains, -0.197);
    int voltRight = closestElectrodeDomain(domains, 0.197);
    int source = closestElectrodeDomain(domains, 0.295);
    assignHomogeneousMaterials(model, new int[]{ground, voltLeft, voltRight, source});

    model.component("comp1").physics("ec").create("term1", "DomainTerminal", 3);
    model.component("comp1").physics("ec").feature("term1").selection().set(source);
    model.component("comp1").physics("ec").feature("term1").set("I0", "I");
    model.component("comp1").physics("ec").create("gnd1", "Ground", 2);
    model.component("comp1").physics("ec").feature("gnd1").selection().set(model.component("comp1").geom("geom1").getAdj(3, 2, ground));
    addAverageCouplings(model, 3, new int[]{voltLeft}, new int[]{voltRight});
    finishSolveSetup(model);

    double[] result = solve(model);
    exportCommonImages(model, scenario, new String[]{"term1", "gnd1"});
    saveAndRecord(out, model, scenario, result, "Old STEP electrode domains kept as metal electrodes");
  }

  private static void runWindow(PrintWriter out) throws IOException {
    String scenario = "window_body_top_faces";
    Model model = baseModel(scenario, false, "");
    double[][] domains = measureDomains(model);
    assignHomogeneousMaterials(model, new int[]{});
    int softBody = largestDomain(domains);
    int bGround = bodyTopFace(model, softBody, -0.295);
    int bVoltLeft = bodyTopFace(model, softBody, -0.197);
    int bVoltRight = bodyTopFace(model, softBody, 0.197);
    int bSource = bodyTopFace(model, softBody, 0.295);

    model.component("comp1").physics("ec").create("term1", "Terminal", 2);
    model.component("comp1").physics("ec").feature("term1").selection().set(bSource);
    model.component("comp1").physics("ec").feature("term1").set("I0", "I");
    model.component("comp1").physics("ec").create("gnd1", "Ground", 2);
    model.component("comp1").physics("ec").feature("gnd1").selection().set(bGround);
    addAverageCouplings(model, 2, new int[]{bVoltLeft}, new int[]{bVoltRight});
    finishSolveSetup(model);

    double[] result = solve(model);
    exportCommonImages(model, scenario, new String[]{"term1", "gnd1"});
    saveAndRecord(out, model, scenario, result, "Boundary Terminal/Ground and boundary voltage windows on body-top faces");
  }

  private static void runPoint(PrintWriter out) throws IOException {
    String scenario = "point_arm_top";
    Model model = baseModel(scenario, true, "");
    assignHomogeneousMaterials(model, new int[]{});
    int pGround = closestPoint(model, -0.295);
    int pVoltLeft = closestPoint(model, -0.197);
    int pVoltRight = closestPoint(model, 0.197);
    int pSource = closestPoint(model, 0.295);

    model.component("comp1").physics("ec").create("pcsSink", "PointCurrentSource", 0);
    model.component("comp1").physics("ec").feature("pcsSink").selection().set(pGround);
    model.component("comp1").physics("ec").feature("pcsSink").set("Qjp", "-I");
    model.component("comp1").physics("ec").create("pcsSource", "PointCurrentSource", 0);
    model.component("comp1").physics("ec").feature("pcsSource").selection().set(pSource);
    model.component("comp1").physics("ec").feature("pcsSource").set("Qjp", "I");
    addAverageCouplings(model, 0, new int[]{pVoltLeft}, new int[]{pVoltRight});
    finishSolveSetup(model);

    double[] result = solve(model);
    exportCommonImages(model, scenario, new String[]{"pcsSink", "pcsSource"});
    saveAndRecord(out, model, scenario, result, "Point current source/sink and point voltage probes on top of arm cylinders");
  }

  private static void runPads(PrintWriter out, String scenario, String padMode, double y, double z) throws IOException {
    Model model = baseModel(scenario, false, padMode);
    double[][] domains = measureDomains(model);
    int ground = closestPadDomain(domains, -0.295, y, z);
    int voltLeft = closestPadDomain(domains, -0.197, y, z);
    int voltRight = closestPadDomain(domains, 0.197, y, z);
    int source = closestPadDomain(domains, 0.295, y, z);
    assignHomogeneousMaterials(model, new int[]{ground, voltLeft, voltRight, source});

    int[] bSource = model.component("comp1").geom("geom1").getAdj(3, 2, source);
    int[] bGround = model.component("comp1").geom("geom1").getAdj(3, 2, ground);
    int[] bVoltLeft = model.component("comp1").geom("geom1").getAdj(3, 2, voltLeft);
    int[] bVoltRight = model.component("comp1").geom("geom1").getAdj(3, 2, voltRight);

    model.component("comp1").physics("ec").create("term1", "Terminal", 2);
    model.component("comp1").physics("ec").feature("term1").selection().set(bSource);
    model.component("comp1").physics("ec").feature("term1").set("I0", "I");
    model.component("comp1").physics("ec").create("gnd1", "Ground", 2);
    model.component("comp1").physics("ec").feature("gnd1").selection().set(bGround);
    addAverageCouplings(model, 2, bVoltLeft, bVoltRight);
    finishSolveSetup(model);

    double[] result = solve(model);
    exportCommonImages(model, scenario, new String[]{"term1", "gnd1"});
    saveAndRecord(out, model, scenario, result, "Visible metal pad electrodes, mode=" + padMode);
  }

  public static void main(String[] args) throws Exception {
    ensureOutDir();
    PrintWriter out = new PrintWriter(new FileWriter(SUMMARY_CSV));
    try {
      out.println("scenario,ZTT_ohm,VR_V,VL_V,notes");
      runOldDomain(out);
      runWindow(out);
      runPoint(out);
      runPads(out, "smallpads_body_top", "body_top", BODY_TOP_Y, BODY_TOP_Z);
      runPads(out, "smallpads_arm_top", "arm_top", ARM_SKIN_Y, ARM_SKIN_Z);
    } finally {
      out.close();
    }
    System.out.println("SUMMARY=" + SUMMARY_CSV);
  }
}
