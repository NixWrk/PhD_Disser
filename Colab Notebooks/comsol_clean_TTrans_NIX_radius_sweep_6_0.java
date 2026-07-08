import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class comsol_clean_TTrans_NIX_radius_sweep_6_0 {
  private static final String STEP_PATH = "C:\\tmp\\TTrans_for_NIX.step";
  private static final String OUT_MPH = "C:\\tmp\\TTrans_NIX_clean_radius_6_0.mph";
  private static final String OUT_SMALLPAD_MPH = "C:\\tmp\\TTrans_NIX_smallpad_radius_6_0.mph";
  private static final String OUT_POINT_MPH = "C:\\tmp\\TTrans_NIX_point_radius_6_0.mph";
  private static final String OUT_WINDOW_MPH = "C:\\tmp\\TTrans_NIX_window_radius_6_0.mph";
  private static final String OUT_SKIN_GAUSSIAN_MPH = "C:\\tmp\\TTrans_NIX_skin_gaussian_radius_6_0.mph";
  private static final String OUT_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_clean_baseline.csv";
  private static final String OUT_SMALLPAD_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_smallpad_baseline.csv";
  private static final String OUT_POINT_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_point_baseline.csv";
  private static final String OUT_WINDOW_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_window_baseline.csv";
  private static final String OUT_POINT_HOM_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_point_homogeneous_baseline.csv";
  private static final String OUT_WINDOW_HOM_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_window_homogeneous_baseline.csv";
  private static final String OUT_WINDOW_RHO10_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_window_rho_div10_baseline.csv";
  private static final String OUT_WINDOW_RHO_MATCH_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_window_rho_match_z1_baseline.csv";
  private static final String OUT_WINDOW_SOFT_MATCH_LUNG_CT_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_window_soft_match_lung_ct_baseline.csv";
  private static final String OUT_WINDOW_SOFT031_LUNG_CT_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_window_soft031_lung_ct_baseline.csv";
  private static final String OUT_SKIN_GAUSSIAN_BASELINE_CSV = "C:\\tmp\\TTrans_NIX_skin_gaussian_baseline.csv";
  private static final String OUT_R0_CSV = "C:\\tmp\\TTrans_NIX_R0_calibration.csv";
  private static final String OUT_DELRH_CSV = "C:\\tmp\\TTrans_NIX_delRh_sweep.csv";
  private static final String OUT_DELRH_QUICK_CSV = "C:\\tmp\\TTrans_NIX_delRh_quick.csv";
  private static final String OUT_SMALLPAD_DELRH_QUICK_CSV = "C:\\tmp\\TTrans_NIX_smallpad_delRh_quick.csv";
  private static final String OUT_POINT_DELRH_QUICK_CSV = "C:\\tmp\\TTrans_NIX_point_delRh_quick.csv";
  private static final String OUT_WINDOW_DELRH_QUICK_CSV = "C:\\tmp\\TTrans_NIX_window_delRh_quick.csv";
  private static final String OUT_ELECTRODE_CSV = "C:\\tmp\\TTrans_NIX_electrode_scan.csv";

  private static final int[] SOFT_DOMAINS = new int[]{2, 7, 8, 9};
  private static final int[] LUNG_DOMAINS = new int[]{4, 10};
  private static final int[] MYOCARDIUM_DOMAINS = new int[]{5};
  private static final int[] BLOOD_DOMAINS = new int[]{6};
  private static final int[] ELECTRODE_DOMAINS = new int[]{1, 3, 11, 12};

  private static final int CURRENT_GROUND_DOMAIN = 1;
  private static final int CURRENT_SOURCE_DOMAIN = 12;
  private static final int VOLTAGE_LEFT_DOMAIN = 3;
  private static final int VOLTAGE_RIGHT_DOMAIN = 11;
  private static final double HEART_CX = -0.026;
  private static final double HEART_CY = 0.0355;
  private static final double HEART_CZ = 0.026;
  private static final double ELECTRODE_CY = -0.050;
  private static final double ELECTRODE_CZ = 0.160;
  private static final double BODY_TOP_Y = -0.026;
  private static final double BODY_TOP_Z = 0.184;
  private static final double ARM_SKIN_Y = -0.050;
  private static final double ARM_SKIN_Z = 0.208;

  private static void material(Model model, String tag, String label, String sigmaExpr, int[] domains) {
    model.component("comp1").material().create(tag, "Common");
    model.component("comp1").material(tag).label(label);
    model.component("comp1").material(tag).selection().set(domains);
    model.component("comp1").material(tag).propertyGroup("def").set("electricconductivity", new String[]{sigmaExpr});
    model.component("comp1").material(tag).propertyGroup("def").set("relpermittivity", new String[]{"1"});
  }

  private static void makeHomogeneousSoft(Model model) {
    String[] sigma = new String[]{"1/rho1_soft"};
    model.component("comp1").material("matLung").propertyGroup("def").set("electricconductivity", sigma);
    model.component("comp1").material("matMyocardium").propertyGroup("def").set("electricconductivity", sigma);
    model.component("comp1").material("matBlood").propertyGroup("def").set("electricconductivity", sigma);
    model.component("comp1").material("matElectrode").propertyGroup("def").set("electricconductivity", sigma);
  }

  private static void divideOwnRhosBy10(Model model) {
    model.param().set("rho1_soft", "0.4728[ohm*m]");
    model.param().set("rho2_lung", "1.7735[ohm*m]");
  }

  private static void setOwnRhosForZ1(Model model) {
    model.param().set("rho1_soft", "0.4337[ohm*m]");
    model.param().set("rho2_lung", "1.6267[ohm*m]");
  }

  private static void setSoftForZ1KeepCtLung(Model model) {
    model.param().set("rho1_soft", "0.5507[ohm*m]");
    model.param().set("rho2_lung", "17.4067[ohm*m]");
  }

  private static void setSoft031KeepCtLung(Model model) {
    model.param().set("rho1_soft", "0.31[ohm*m]");
    model.param().set("rho2_lung", "17.4067[ohm*m]");
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

  private static double bx(double[] b) {
    return 0.5 * (b[0] + b[1]);
  }

  private static double bdx(double[] b) {
    return b[1] - b[0];
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

  private static double heartCenterDistance(double[] d) {
    double x = cx(d) - HEART_CX;
    double y = cy(d) - HEART_CY;
    double z = cz(d) - HEART_CZ;
    return Math.sqrt(x * x + y * y + z * z);
  }

  private static int closestElectrode(double[][] domains, double targetX) {
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
      throw new RuntimeException("Could not classify electrode near x=" + targetX);
    }
    return best;
  }

  private static int closestSmallPad(double[][] domains, double targetX) {
    int best = -1;
    double bestDist = Double.POSITIVE_INFINITY;
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      double x = cx(d) - targetX;
      double y = cy(d) - BODY_TOP_Y;
      double z = cz(d) - BODY_TOP_Z;
      double dist = Math.sqrt(x * x + y * y + z * z);
      if (dx(d) < 0.02 && volume(d) > 1.0e-8 && volume(d) < 3.0e-6 && dist < bestDist) {
        best = domainId(d);
        bestDist = dist;
      }
    }
    if (best < 0 || bestDist > 0.02) {
      throw new RuntimeException("Could not classify small electrode pad near x=" + targetX + ", bestDist=" + bestDist);
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

  private static double[] domainById(double[][] domains, int id) {
    for (int i = 0; i < domains.length; i++) {
      if (domainId(domains[i]) == id) {
        return domains[i];
      }
    }
    throw new RuntimeException("Missing domain id=" + id);
  }

  private static int inwardFace(Model model, double[][] domains, int domain) {
    double[] d = domainById(domains, domain);
    double targetX = cx(d) >= 0.0 ? d[2] : d[3];
    return faceByX(model, domain, targetX);
  }

  private static int outwardFace(Model model, double[][] domains, int domain) {
    double[] d = domainById(domains, domain);
    double targetX = cx(d) >= 0.0 ? d[3] : d[2];
    return faceByX(model, domain, targetX);
  }

  private static int faceByX(Model model, int domain, double targetX) {
    int[] boundaries = model.component("comp1").geom("geom1").getAdj(3, 2, domain);
    int best = -1;
    double bestScore = Double.POSITIVE_INFINITY;
    double[] bestBbox = null;
    for (int i = 0; i < boundaries.length; i++) {
      int b = boundaries[i];
      model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 2);
      model.component("comp1").geom("geom1").measureFinal().selection().set(b);
      double[] bbox = model.component("comp1").geom("geom1").measureFinal().getBoundingBox();
      double yPenalty = intervalDistance(ELECTRODE_CY, bbox[2], bbox[3]);
      double zPenalty = intervalDistance(ELECTRODE_CZ, bbox[4], bbox[5]);
      double yzArea = Math.max(0.0, bbox[3] - bbox[2]) * Math.max(0.0, bbox[5] - bbox[4]);
      double score = Math.abs(bx(bbox) - targetX) + 100.0 * bdx(bbox) + 100.0 * yPenalty + 100.0 * zPenalty - 1.0e-3 * yzArea;
      if (score < bestScore) {
        best = b;
        bestScore = score;
        bestBbox = bbox;
      }
    }
    if (best < 0) {
      throw new RuntimeException("Could not find x-face for domain=" + domain + ", targetX=" + targetX);
    }
    System.out.println("FACE_PICK,domain=" + domain + ",targetX=" + targetX + ",boundary=" + best + ",bbox=" + Arrays.toString(bestBbox) + ",score=" + bestScore);
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
    System.out.println("BODY_TOP_FACE,targetX=" + targetX + ",boundary=" + best + ",bbox=" + Arrays.toString(bestBbox) + ",score=" + bestScore);
    return best;
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

  private static double center(double lo, double hi) {
    return 0.5 * (lo + hi);
  }

  private static boolean contains(List<Integer> values, int id) {
    for (Integer value : values) {
      if (value.intValue() == id) {
        return true;
      }
    }
    return false;
  }

  private static int[] toIntArray(List<Integer> values) {
    int[] result = new int[values.size()];
    for (int i = 0; i < values.size(); i++) {
      result[i] = values.get(i).intValue();
    }
    Arrays.sort(result);
    return result;
  }

  private static void updateDomainSelections(Model model, double heartRadiusMm) {
    double[][] domains = measureDomains(model);
    int ground = closestElectrode(domains, -0.295);
    int voltLeft = closestElectrode(domains, -0.197);
    int voltRight = closestElectrode(domains, 0.197);
    int source = closestElectrode(domains, 0.295);

    double[] myocardiumDomain = null;
    double[] bloodDomain = null;
    double expectedMyocardiumDx = 2.0 * heartRadiusMm / 1000.0;
    double expectedBloodDx = expectedMyocardiumDx * (65.0 / 85.0);
    double bestMyocardiumScore = Double.POSITIVE_INFINITY;
    double bestBloodScore = Double.POSITIVE_INFINITY;
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < 0.16 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double myocardiumScore = Math.abs(dx(d) - expectedMyocardiumDx);
        if (myocardiumScore < bestMyocardiumScore - 1.0e-6 ||
            (Math.abs(myocardiumScore - bestMyocardiumScore) <= 1.0e-6 && myocardiumDomain != null && volume(d) > volume(myocardiumDomain))) {
          myocardiumDomain = d;
          bestMyocardiumScore = myocardiumScore;
        }
      }
    }
    if (myocardiumDomain == null) {
      throw new RuntimeException("Could not classify myocardium domain");
    }
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (domainId(d) == domainId(myocardiumDomain)) {
        continue;
      }
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < dx(myocardiumDomain) * 0.98 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double bloodScore = Math.abs(dx(d) - expectedBloodDx);
        if (bloodScore < bestBloodScore - 1.0e-6 ||
            (Math.abs(bloodScore - bestBloodScore) <= 1.0e-6 && bloodDomain != null && volume(d) > volume(bloodDomain))) {
          bloodDomain = d;
          bestBloodScore = bloodScore;
        }
      }
    }
    if (myocardiumDomain == null || bloodDomain == null) {
      throw new RuntimeException("Could not classify myocardium and blood domains");
    }
    int myocardium = domainId(myocardiumDomain);
    int blood = domainId(bloodDomain);

    List<Integer> used = new ArrayList<Integer>();
    used.add(myocardium);
    used.add(blood);

    List<Integer> lungs = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      int id = domainId(d);
      if (!contains(used, id) && volume(d) > 1.0e-3 && volume(d) < 8.0e-3) {
        lungs.add(Integer.valueOf(id));
      }
    }
    if (lungs.size() != 2) {
      throw new RuntimeException("Expected two lung domains, classified " + lungs.size());
    }
    used.addAll(lungs);

    List<Integer> soft = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      int id = domainId(domains[i]);
      if (!contains(used, id)) {
        soft.add(Integer.valueOf(id));
      }
    }

    int[] softDomains = toIntArray(soft);
    int[] lungDomains = toIntArray(lungs);
    model.component("comp1").material("matSoft").selection().set(softDomains);
    model.component("comp1").material("matLung").selection().set(lungDomains);
    model.component("comp1").material("matMyocardium").selection().set(myocardium);
    model.component("comp1").material("matBlood").selection().set(blood);
    model.component("comp1").material("matElectrode").selection().set(new int[]{ground, voltLeft, voltRight, source});

    model.component("comp1").physics("ec").feature("term1").selection().set(source);
    int[] groundBoundaries = model.component("comp1").geom("geom1").getAdj(3, 2, ground);
    model.component("comp1").physics("ec").feature("gnd1").selection().set(groundBoundaries);
    model.component("comp1").cpl("aveVL").selection().set(voltLeft);
    model.component("comp1").cpl("aveVR").selection().set(voltRight);

    System.out.println(
      "CLASSIFY,soft=" + Arrays.toString(softDomains) +
      ",lungs=" + Arrays.toString(lungDomains) +
      ",myocardium=" + myocardium +
      ",blood=" + blood +
      ",heartRadiusMm=" + heartRadiusMm +
      ",electrodes=" + Arrays.toString(new int[]{ground, voltLeft, voltRight, source}) +
      ",groundBoundaries=" + Arrays.toString(groundBoundaries)
    );
  }

  private static void updateSmallPadSelections(Model model, double heartRadiusMm) {
    double[][] domains = measureDomains(model);
    int ground = closestSmallPad(domains, -0.295);
    int voltLeft = closestSmallPad(domains, -0.197);
    int voltRight = closestSmallPad(domains, 0.197);
    int source = closestSmallPad(domains, 0.295);

    double[] myocardiumDomain = null;
    double[] bloodDomain = null;
    double expectedMyocardiumDx = 2.0 * heartRadiusMm / 1000.0;
    double expectedBloodDx = expectedMyocardiumDx * (65.0 / 85.0);
    double bestMyocardiumScore = Double.POSITIVE_INFINITY;
    double bestBloodScore = Double.POSITIVE_INFINITY;
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < 0.16 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double myocardiumScore = Math.abs(dx(d) - expectedMyocardiumDx);
        if (myocardiumScore < bestMyocardiumScore - 1.0e-6 ||
            (Math.abs(myocardiumScore - bestMyocardiumScore) <= 1.0e-6 && myocardiumDomain != null && volume(d) > volume(myocardiumDomain))) {
          myocardiumDomain = d;
          bestMyocardiumScore = myocardiumScore;
        }
      }
    }
    if (myocardiumDomain == null) {
      throw new RuntimeException("Could not classify myocardium domain for small-pad model");
    }
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (domainId(d) == domainId(myocardiumDomain)) {
        continue;
      }
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < dx(myocardiumDomain) * 0.98 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double bloodScore = Math.abs(dx(d) - expectedBloodDx);
        if (bloodScore < bestBloodScore - 1.0e-6 ||
            (Math.abs(bloodScore - bestBloodScore) <= 1.0e-6 && bloodDomain != null && volume(d) > volume(bloodDomain))) {
          bloodDomain = d;
          bestBloodScore = bloodScore;
        }
      }
    }
    if (bloodDomain == null) {
      throw new RuntimeException("Could not classify blood domain for small-pad model");
    }
    int myocardium = domainId(myocardiumDomain);
    int blood = domainId(bloodDomain);

    List<Integer> used = new ArrayList<Integer>();
    used.add(myocardium);
    used.add(blood);

    List<Integer> lungs = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      int id = domainId(d);
      if (!contains(used, id) && volume(d) > 1.0e-3 && volume(d) < 8.0e-3) {
        lungs.add(Integer.valueOf(id));
      }
    }
    if (lungs.size() != 2) {
      throw new RuntimeException("Expected two lung domains in small-pad model, classified " + lungs.size());
    }
    used.addAll(lungs);

    List<Integer> soft = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      int id = domainId(domains[i]);
      if (!contains(used, id)) {
        soft.add(Integer.valueOf(id));
      }
    }

    int[] softDomains = toIntArray(soft);
    int[] lungDomains = toIntArray(lungs);
    model.component("comp1").material("matSoft").selection().set(softDomains);
    model.component("comp1").material("matLung").selection().set(lungDomains);
    model.component("comp1").material("matMyocardium").selection().set(myocardium);
    model.component("comp1").material("matBlood").selection().set(blood);
    model.component("comp1").material("matElectrode").selection().set(new int[]{});

    int[] sourceBoundaries = model.component("comp1").geom("geom1").getAdj(3, 2, source);
    model.component("comp1").physics("ec").feature("term1").selection().set(sourceBoundaries);
    int[] groundBoundaries = model.component("comp1").geom("geom1").getAdj(3, 2, ground);
    model.component("comp1").physics("ec").feature("gnd1").selection().set(groundBoundaries);
    int[] voltLeftBoundaries = model.component("comp1").geom("geom1").getAdj(3, 2, voltLeft);
    int[] voltRightBoundaries = model.component("comp1").geom("geom1").getAdj(3, 2, voltRight);
    model.component("comp1").cpl("aveVL").selection().set(voltLeftBoundaries);
    model.component("comp1").cpl("aveVR").selection().set(voltRightBoundaries);

    System.out.println(
      "CLASSIFY_SMALLPAD,soft=" + Arrays.toString(softDomains) +
      ",lungs=" + Arrays.toString(lungDomains) +
      ",myocardium=" + myocardium +
      ",blood=" + blood +
      ",heartRadiusMm=" + heartRadiusMm +
      ",pads=" + Arrays.toString(new int[]{ground, voltLeft, voltRight, source}) +
      ",sourceBoundaries=" + Arrays.toString(sourceBoundaries) +
      ",voltLeftBoundaries=" + Arrays.toString(voltLeftBoundaries) +
      ",voltRightBoundaries=" + Arrays.toString(voltRightBoundaries) +
      ",groundBoundaries=" + Arrays.toString(groundBoundaries)
    );
  }

  private static void updatePointSelections(Model model, double heartRadiusMm) {
    double[][] domains = measureDomains(model);
    int pGround = closestPoint(model, -0.295);
    int pVoltLeft = closestPoint(model, -0.197);
    int pVoltRight = closestPoint(model, 0.197);
    int pSource = closestPoint(model, 0.295);

    double[] myocardiumDomain = null;
    double[] bloodDomain = null;
    double expectedMyocardiumDx = 2.0 * heartRadiusMm / 1000.0;
    double expectedBloodDx = expectedMyocardiumDx * (65.0 / 85.0);
    double bestMyocardiumScore = Double.POSITIVE_INFINITY;
    double bestBloodScore = Double.POSITIVE_INFINITY;
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < 0.16 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double myocardiumScore = Math.abs(dx(d) - expectedMyocardiumDx);
        if (myocardiumScore < bestMyocardiumScore - 1.0e-6 ||
            (Math.abs(myocardiumScore - bestMyocardiumScore) <= 1.0e-6 && myocardiumDomain != null && volume(d) > volume(myocardiumDomain))) {
          myocardiumDomain = d;
          bestMyocardiumScore = myocardiumScore;
        }
      }
    }
    if (myocardiumDomain == null) {
      throw new RuntimeException("Could not classify myocardium domain for point model");
    }
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (domainId(d) == domainId(myocardiumDomain)) {
        continue;
      }
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < dx(myocardiumDomain) * 0.98 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double bloodScore = Math.abs(dx(d) - expectedBloodDx);
        if (bloodScore < bestBloodScore - 1.0e-6 ||
            (Math.abs(bloodScore - bestBloodScore) <= 1.0e-6 && bloodDomain != null && volume(d) > volume(bloodDomain))) {
          bloodDomain = d;
          bestBloodScore = bloodScore;
        }
      }
    }
    if (bloodDomain == null) {
      throw new RuntimeException("Could not classify blood domain for point model");
    }
    int myocardium = domainId(myocardiumDomain);
    int blood = domainId(bloodDomain);

    List<Integer> used = new ArrayList<Integer>();
    used.add(myocardium);
    used.add(blood);

    List<Integer> lungs = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      int id = domainId(d);
      if (!contains(used, id) && volume(d) > 1.0e-3 && volume(d) < 8.0e-3) {
        lungs.add(Integer.valueOf(id));
      }
    }
    if (lungs.size() != 2) {
      throw new RuntimeException("Expected two lung domains in point model, classified " + lungs.size());
    }
    used.addAll(lungs);

    List<Integer> soft = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      int id = domainId(domains[i]);
      if (!contains(used, id)) {
        soft.add(Integer.valueOf(id));
      }
    }

    int[] softDomains = toIntArray(soft);
    int[] lungDomains = toIntArray(lungs);
    model.component("comp1").material("matSoft").selection().set(softDomains);
    model.component("comp1").material("matLung").selection().set(lungDomains);
    model.component("comp1").material("matMyocardium").selection().set(myocardium);
    model.component("comp1").material("matBlood").selection().set(blood);
    model.component("comp1").material("matElectrode").selection().set(new int[]{});

    model.component("comp1").physics("ec").feature("pcsSink").selection().set(pGround);
    model.component("comp1").physics("ec").feature("pcsSource").selection().set(pSource);
    model.component("comp1").cpl("aveVL").selection().set(pVoltLeft);
    model.component("comp1").cpl("aveVR").selection().set(pVoltRight);

    System.out.println(
      "CLASSIFY_POINT,soft=" + Arrays.toString(softDomains) +
      ",lungs=" + Arrays.toString(lungDomains) +
      ",myocardium=" + myocardium +
      ",blood=" + blood +
      ",heartRadiusMm=" + heartRadiusMm +
      ",points=" + Arrays.toString(new int[]{pGround, pVoltLeft, pVoltRight, pSource})
    );
  }

  private static void updateWindowSelections(Model model, double heartRadiusMm) {
    double[][] domains = measureDomains(model);
    int softBody = largestDomain(domains);

    double[] myocardiumDomain = null;
    double[] bloodDomain = null;
    double expectedMyocardiumDx = 2.0 * heartRadiusMm / 1000.0;
    double expectedBloodDx = expectedMyocardiumDx * (65.0 / 85.0);
    double bestMyocardiumScore = Double.POSITIVE_INFINITY;
    double bestBloodScore = Double.POSITIVE_INFINITY;
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < 0.16 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double myocardiumScore = Math.abs(dx(d) - expectedMyocardiumDx);
        if (myocardiumScore < bestMyocardiumScore - 1.0e-6 ||
            (Math.abs(myocardiumScore - bestMyocardiumScore) <= 1.0e-6 && myocardiumDomain != null && volume(d) > volume(myocardiumDomain))) {
          myocardiumDomain = d;
          bestMyocardiumScore = myocardiumScore;
        }
      }
    }
    if (myocardiumDomain == null) {
      throw new RuntimeException("Could not classify myocardium domain for window model");
    }
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (domainId(d) == domainId(myocardiumDomain)) {
        continue;
      }
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < dx(myocardiumDomain) * 0.98 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double bloodScore = Math.abs(dx(d) - expectedBloodDx);
        if (bloodScore < bestBloodScore - 1.0e-6 ||
            (Math.abs(bloodScore - bestBloodScore) <= 1.0e-6 && bloodDomain != null && volume(d) > volume(bloodDomain))) {
          bloodDomain = d;
          bestBloodScore = bloodScore;
        }
      }
    }
    if (bloodDomain == null) {
      throw new RuntimeException("Could not classify blood domain for window model");
    }
    int myocardium = domainId(myocardiumDomain);
    int blood = domainId(bloodDomain);

    List<Integer> used = new ArrayList<Integer>();
    used.add(myocardium);
    used.add(blood);

    List<Integer> lungs = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      int id = domainId(d);
      if (!contains(used, id) && volume(d) > 1.0e-3 && volume(d) < 8.0e-3) {
        lungs.add(Integer.valueOf(id));
      }
    }
    if (lungs.size() != 2) {
      throw new RuntimeException("Expected two lung domains in window model, classified " + lungs.size());
    }
    used.addAll(lungs);

    List<Integer> soft = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      int id = domainId(domains[i]);
      if (!contains(used, id)) {
        soft.add(Integer.valueOf(id));
      }
    }

    int[] softDomains = toIntArray(soft);
    int[] lungDomains = toIntArray(lungs);
    model.component("comp1").material("matSoft").selection().set(softDomains);
    model.component("comp1").material("matLung").selection().set(lungDomains);
    model.component("comp1").material("matMyocardium").selection().set(myocardium);
    model.component("comp1").material("matBlood").selection().set(blood);
    model.component("comp1").material("matElectrode").selection().set(new int[]{});

    int bGround = bodyTopFace(model, softBody, -0.295);
    int bVoltLeft = bodyTopFace(model, softBody, -0.197);
    int bVoltRight = bodyTopFace(model, softBody, 0.197);
    int bSource = bodyTopFace(model, softBody, 0.295);
    model.component("comp1").physics("ec").feature("term1").selection().set(bSource);
    model.component("comp1").physics("ec").feature("gnd1").selection().set(bGround);
    model.component("comp1").cpl("aveVL").selection().set(bVoltLeft);
    model.component("comp1").cpl("aveVR").selection().set(bVoltRight);

    System.out.println(
      "CLASSIFY_WINDOW,soft=" + Arrays.toString(softDomains) +
      ",lungs=" + Arrays.toString(lungDomains) +
      ",myocardium=" + myocardium +
      ",blood=" + blood +
      ",heartRadiusMm=" + heartRadiusMm +
      ",softBody=" + softBody +
      ",faces=" + Arrays.toString(new int[]{bGround, bVoltLeft, bVoltRight, bSource})
    );
  }

  private static int[] updateNoElectrodeMaterials(Model model, double heartRadiusMm, String label) {
    double[][] domains = measureDomains(model);
    int softBody = largestDomain(domains);

    double[] myocardiumDomain = null;
    double[] bloodDomain = null;
    double expectedMyocardiumDx = 2.0 * heartRadiusMm / 1000.0;
    double expectedBloodDx = expectedMyocardiumDx * (65.0 / 85.0);
    double bestMyocardiumScore = Double.POSITIVE_INFINITY;
    double bestBloodScore = Double.POSITIVE_INFINITY;
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < 0.16 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double myocardiumScore = Math.abs(dx(d) - expectedMyocardiumDx);
        if (myocardiumScore < bestMyocardiumScore - 1.0e-6 ||
            (Math.abs(myocardiumScore - bestMyocardiumScore) <= 1.0e-6 && myocardiumDomain != null && volume(d) > volume(myocardiumDomain))) {
          myocardiumDomain = d;
          bestMyocardiumScore = myocardiumScore;
        }
      }
    }
    if (myocardiumDomain == null) {
      throw new RuntimeException("Could not classify myocardium domain for " + label);
    }
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      if (domainId(d) == domainId(myocardiumDomain)) {
        continue;
      }
      if (heartCenterDistance(d) < 0.012 && dx(d) > 0.03 && dx(d) < dx(myocardiumDomain) * 0.98 && volume(d) > 1.0e-5 && volume(d) < 1.0e-3) {
        double bloodScore = Math.abs(dx(d) - expectedBloodDx);
        if (bloodScore < bestBloodScore - 1.0e-6 ||
            (Math.abs(bloodScore - bestBloodScore) <= 1.0e-6 && bloodDomain != null && volume(d) > volume(bloodDomain))) {
          bloodDomain = d;
          bestBloodScore = bloodScore;
        }
      }
    }
    if (bloodDomain == null) {
      throw new RuntimeException("Could not classify blood domain for " + label);
    }
    int myocardium = domainId(myocardiumDomain);
    int blood = domainId(bloodDomain);

    List<Integer> used = new ArrayList<Integer>();
    used.add(myocardium);
    used.add(blood);

    List<Integer> lungs = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      int id = domainId(d);
      if (!contains(used, id) && volume(d) > 1.0e-3 && volume(d) < 8.0e-3) {
        lungs.add(Integer.valueOf(id));
      }
    }
    if (lungs.size() != 2) {
      throw new RuntimeException("Expected two lung domains in " + label + ", classified " + lungs.size());
    }
    used.addAll(lungs);

    List<Integer> soft = new ArrayList<Integer>();
    for (int i = 0; i < domains.length; i++) {
      int id = domainId(domains[i]);
      if (!contains(used, id)) {
        soft.add(Integer.valueOf(id));
      }
    }

    int[] softDomains = toIntArray(soft);
    int[] lungDomains = toIntArray(lungs);
    model.component("comp1").material("matSoft").selection().set(softDomains);
    model.component("comp1").material("matLung").selection().set(lungDomains);
    model.component("comp1").material("matMyocardium").selection().set(myocardium);
    model.component("comp1").material("matBlood").selection().set(blood);
    model.component("comp1").material("matElectrode").selection().set(new int[]{});

    int[] softBodyBoundaries = model.component("comp1").geom("geom1").getAdj(3, 2, softBody);
    System.out.println(
      "CLASSIFY_" + label + ",soft=" + Arrays.toString(softDomains) +
      ",lungs=" + Arrays.toString(lungDomains) +
      ",myocardium=" + myocardium +
      ",blood=" + blood +
      ",heartRadiusMm=" + heartRadiusMm +
      ",softBody=" + softBody +
      ",softBodyBoundaries=" + Arrays.toString(softBodyBoundaries)
    );
    return softBodyBoundaries;
  }

  private static void updateSkinGaussianSelections(Model model, double heartRadiusMm) {
    int[] boundaries = updateNoElectrodeMaterials(model, heartRadiusMm, "SKIN_GAUSSIAN");
    model.component("comp1").physics("ec").feature("ncdSink").selection().set(boundaries);
    model.component("comp1").physics("ec").feature("ncdSource").selection().set(boundaries);
    model.component("comp1").cpl("intA").selection().set(boundaries);
    model.component("comp1").cpl("intM").selection().set(boundaries);
    model.component("comp1").cpl("intN").selection().set(boundaries);
    model.component("comp1").cpl("intB").selection().set(boundaries);
  }

  private static void printDomainMap(Model model, String label) {
    double[][] domains = measureDomains(model);
    for (int i = 0; i < domains.length; i++) {
      double[] d = domains[i];
      System.out.println(
        "DOMAIN_MAP," + label +
        ",id=" + domainId(d) +
        ",vol=" + volume(d) +
        ",bbox=[" + d[2] + "," + d[3] + "," + d[4] + "," + d[5] + "," + d[6] + "," + d[7] + "]" +
        ",center=[" + cx(d) + "," + cy(d) + "," + cz(d) + "]" +
        ",dx=" + dx(d)
      );
    }
  }

  private static void mapDelRh(Model model, double rh0mm) {
    double[] deltas = new double[]{-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0};
    for (int i = 0; i < deltas.length; i++) {
      double delRh = deltas[i];
      model.param().set("Rh0", rh0mm + "[mm]");
      model.param().set("delRh", delRh + "[mm]");
      model.component("comp1").geom("geom1").run();
      updateDomainSelections(model, rh0mm - delRh);
      printDomainMap(model, "Rh0=" + rh0mm + ";delRh=" + delRh);
    }
  }

  private static void createSmallPad(Model model, String tag, double x) {
    model.component("comp1").geom("geom1").create(tag, "Sphere");
    model.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{x, BODY_TOP_Y, BODY_TOP_Z});
    model.component("comp1").geom("geom1").feature(tag).set("r", "3[mm]");
  }

  private static void createElectrodePoint(Model model, String tag, double x) {
    model.component("comp1").geom("geom1").create(tag, "Point");
    model.component("comp1").geom("geom1").feature(tag).set("p", new double[]{x, ARM_SKIN_Y, ARM_SKIN_Z});
  }

  private static Model buildModel() {
    return buildModel(false);
  }

  private static Model buildModel(boolean smallPads) {
    return buildModel(smallPads, false);
  }

  private static Model buildModel(boolean smallPads, boolean pointElectrodes) {
    return buildModel(smallPads, pointElectrodes, false);
  }

  private static Model buildModel(boolean smallPads, boolean pointElectrodes, boolean windowElectrodes) {
    return buildModel(smallPads, pointElectrodes, windowElectrodes, false);
  }

  private static Model buildModel(boolean smallPads, boolean pointElectrodes, boolean windowElectrodes, boolean skinGaussianElectrodes) {
    Model model = ModelUtil.create("TTransNIXRadius");
    model.modelPath("C:\\tmp");
    model.label(skinGaussianElectrodes ? "TTrans_NIX_skin_gaussian_radius_6_0.mph" : (windowElectrodes ? "TTrans_NIX_window_radius_6_0.mph" : (pointElectrodes ? "TTrans_NIX_point_radius_6_0.mph" : (smallPads ? "TTrans_NIX_smallpad_radius_6_0.mph" : "TTrans_NIX_clean_radius_6_0.mph"))));

    model.param().set("I", "1[A]");
    model.param().set("Z1_target", "12.703[ohm]");
    model.param().set("Z2_target", "12.626[ohm]");
    model.param().set("rho1_soft", "4.728[ohm*m]");
    model.param().set("rho2_lung", "17.735[ohm*m]");
    model.param().set("rho_myocardium", "1/0.215[S/m]");
    model.param().set("rho_blood", "1/0.703[S/m]");
    model.param().set("rho_bone_like", "1/0.0208[S/m]");
    model.param().set("sigma_electrode", "4.032e6[S/m]");
    model.param().set("Rh_geom", "42.5[mm]");
    model.param().set("Rh0", "42.5[mm]");
    model.param().set("delRh", "0[mm]");
    model.param().set("heartRadius", "Rh0-delRh");
    model.param().set("heartScale", "(Rh0-delRh)/Rh_geom");
    model.param().set("fillOuterRadius", "max(Rh_geom,heartRadius)");
    model.param().set("yc_elec", "-26[mm]");
    model.param().set("zc_elec", "184[mm]");
    model.param().set("r_win", "5[mm]");
    model.param().set("y_arm", "-50[mm]");
    model.param().set("z_arm_top", "208[mm]");
    model.param().set("r_skin", "8[mm]");
    model.param().set("xA", "-295[mm]");
    model.param().set("xM", "-197[mm]");
    model.param().set("xN", "197[mm]");
    model.param().set("xB", "295[mm]");

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
    model.component("comp1").geom("geom1").feature("scaHeart").set("pos", new double[]{-0.026, 0.0355, 0.026});

    model.component("comp1").geom("geom1").create("sphFillOuter", "Sphere");
    model.component("comp1").geom("geom1").feature("sphFillOuter").set("pos", new double[]{HEART_CX, HEART_CY, HEART_CZ});
    model.component("comp1").geom("geom1").feature("sphFillOuter").set("r", "fillOuterRadius");
    model.component("comp1").geom("geom1").create("sphFillInner", "Sphere");
    model.component("comp1").geom("geom1").feature("sphFillInner").set("pos", new double[]{HEART_CX, HEART_CY, HEART_CZ});
    model.component("comp1").geom("geom1").feature("sphFillInner").set("r", "heartRadius");
    model.component("comp1").geom("geom1").create("difSoftFill", "Difference");
    model.component("comp1").geom("geom1").feature("difSoftFill").selection("input").set("sphFillOuter");
    model.component("comp1").geom("geom1").feature("difSoftFill").selection("input2").set("sphFillInner");
    if (smallPads) {
      createSmallPad(model, "padA_ground", -0.295);
      createSmallPad(model, "padM_left", -0.197);
      createSmallPad(model, "padN_right", 0.197);
      createSmallPad(model, "padB_source", 0.295);
    }
    if (pointElectrodes) {
      createElectrodePoint(model, "ptA_ground", -0.295);
      createElectrodePoint(model, "ptM_left", -0.197);
      createElectrodePoint(model, "ptN_right", 0.197);
      createElectrodePoint(model, "ptB_source", 0.295);
    }
    model.component("comp1").geom("geom1").run();

    material(model, "matSoft", "soft tissue from rho1", "1/rho1_soft", SOFT_DOMAINS);
    material(model, "matLung", "lung from rho2", "1/rho2_lung", LUNG_DOMAINS);
    material(model, "matMyocardium", "myocardium literature", "1/rho_myocardium", MYOCARDIUM_DOMAINS);
    material(model, "matBlood", "blood literature", "1/rho_blood", BLOOD_DOMAINS);
    material(model, "matElectrode", "steel electrodes", "sigma_electrode", ELECTRODE_DOMAINS);

    model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
    if (skinGaussianElectrodes) {
      model.component("comp1").physics("ec").create("ncdSink", "NormalCurrentDensity", 2);
      model.component("comp1").physics("ec").feature("ncdSink").set("nJ", "-I*wA/intA(wA)");
      model.component("comp1").physics("ec").feature("ncdSink").label("Gaussian current sink on left arm skin");
      model.component("comp1").physics("ec").create("ncdSource", "NormalCurrentDensity", 2);
      model.component("comp1").physics("ec").feature("ncdSource").set("nJ", "I*wB/intB(wB)");
      model.component("comp1").physics("ec").feature("ncdSource").label("Gaussian current source on right arm skin");
    } else if (windowElectrodes) {
      model.component("comp1").physics("ec").create("term1", "Terminal", 2);
      model.component("comp1").physics("ec").feature("term1").set("I0", "I");
      model.component("comp1").physics("ec").feature("term1").label("Current source on body top surface");
      model.component("comp1").physics("ec").create("gnd1", "Ground", 2);
    } else if (pointElectrodes) {
      model.component("comp1").physics("ec").create("pcsSink", "PointCurrentSource", 0);
      model.component("comp1").physics("ec").feature("pcsSink").set("Qjp", "-I");
      model.component("comp1").physics("ec").create("pcsSource", "PointCurrentSource", 0);
      model.component("comp1").physics("ec").feature("pcsSource").set("Qjp", "I");
    } else if (smallPads) {
      model.component("comp1").physics("ec").create("term1", "Terminal", 2);
      model.component("comp1").physics("ec").feature("term1").selection().set(model.component("comp1").geom("geom1").getAdj(3, 2, CURRENT_SOURCE_DOMAIN));
    } else {
      model.component("comp1").physics("ec").create("term1", "DomainTerminal", 3);
      model.component("comp1").physics("ec").feature("term1").selection().set(CURRENT_SOURCE_DOMAIN);
    }
    if (!pointElectrodes && !windowElectrodes && !skinGaussianElectrodes) {
      model.component("comp1").physics("ec").feature("term1").set("I0", "I");
      model.component("comp1").physics("ec").feature("term1").label("Current source, right outer electrode");
    }

    if (!pointElectrodes && !windowElectrodes && !skinGaussianElectrodes) {
      int[] groundBoundaries = model.component("comp1").geom("geom1").getAdj(3, 2, CURRENT_GROUND_DOMAIN);
      System.out.println("GROUND_BOUNDARIES=" + Arrays.toString(groundBoundaries));
      model.component("comp1").physics("ec").create("gnd1", "Ground", 2);
      model.component("comp1").physics("ec").feature("gnd1").selection().set(groundBoundaries);
    }

    if (skinGaussianElectrodes) {
      model.component("comp1").cpl().create("intA", "Integration");
      model.component("comp1").cpl("intA").selection().geom("geom1", 2);
      model.component("comp1").cpl().create("intM", "Integration");
      model.component("comp1").cpl("intM").selection().geom("geom1", 2);
      model.component("comp1").cpl().create("intN", "Integration");
      model.component("comp1").cpl("intN").selection().geom("geom1", 2);
      model.component("comp1").cpl().create("intB", "Integration");
      model.component("comp1").cpl("intB").selection().geom("geom1", 2);
    } else {
      model.component("comp1").cpl().create("aveVL", "Average");
      model.component("comp1").cpl("aveVL").selection().geom("geom1", pointElectrodes ? 0 : ((smallPads || windowElectrodes) ? 2 : 3));
      model.component("comp1").cpl("aveVL").selection().set(VOLTAGE_LEFT_DOMAIN);
      model.component("comp1").cpl().create("aveVR", "Average");
      model.component("comp1").cpl("aveVR").selection().geom("geom1", pointElectrodes ? 0 : ((smallPads || windowElectrodes) ? 2 : 3));
      model.component("comp1").cpl("aveVR").selection().set(VOLTAGE_RIGHT_DOMAIN);
    }

    if (skinGaussianElectrodes) {
      updateSkinGaussianSelections(model, 42.5);
    } else if (windowElectrodes) {
      updateWindowSelections(model, 42.5);
    } else if (pointElectrodes) {
      updatePointSelections(model, 42.5);
    } else if (smallPads) {
      updateSmallPadSelections(model, 42.5);
    } else {
      updateDomainSelections(model, 42.5);
    }

    model.component("comp1").variable().create("var1");
    model.component("comp1").variable("var1").set("w_elec", "exp(-((y-yc_elec)^2+(z-zc_elec)^2)/r_win^2)");
    if (skinGaussianElectrodes) {
      model.component("comp1").variable("var1").set("wA", "exp(-((x-xA)^2+(y-y_arm)^2+(z-z_arm_top)^2)/r_skin^2)");
      model.component("comp1").variable("var1").set("wM", "exp(-((x-xM)^2+(y-y_arm)^2+(z-z_arm_top)^2)/r_skin^2)");
      model.component("comp1").variable("var1").set("wN", "exp(-((x-xN)^2+(y-y_arm)^2+(z-z_arm_top)^2)/r_skin^2)");
      model.component("comp1").variable("var1").set("wB", "exp(-((x-xB)^2+(y-y_arm)^2+(z-z_arm_top)^2)/r_skin^2)");
      model.component("comp1").variable("var1").set("VL", "intM(V*wM)/intM(wM)");
      model.component("comp1").variable("var1").set("VR", "intN(V*wN)/intN(wN)");
    } else {
      model.component("comp1").variable("var1").set("VL", "aveVL(V)");
      model.component("comp1").variable("var1").set("VR", "aveVR(V)");
    }
    model.component("comp1").variable("var1").set("ZTT", "abs(VR-VL)/I");
    model.component("comp1").variable("var1").set("dZ_from_Z1", "ZTT-Z1_target");
    model.component("comp1").variable("var1").set("dZ_from_Z2", "ZTT-Z2_target");

    model.component("comp1").mesh("mesh1").autoMeshSize(5);
    model.component("comp1").mesh("mesh1").run();

    model.study().create("std1");
    model.study("std1").create("stat", "Stationary");

    model.result().numerical().create("gevZ", "EvalGlobal");
    model.result().numerical("gevZ").set("expr", new String[]{"ZTT", "VR", "VL", "dZ_from_Z1", "dZ_from_Z2"});
    model.result().numerical("gevZ").set("unit", new String[]{"ohm", "V", "V", "ohm", "ohm"});
    return model;
  }

  private static double[] solve(Model model, double rh0mm, double delRhmm) {
    return solve(model, rh0mm, delRhmm, false);
  }

  private static double[] solve(Model model, double rh0mm, double delRhmm, boolean smallPads) {
    return solve(model, rh0mm, delRhmm, smallPads, false);
  }

  private static double[] solve(Model model, double rh0mm, double delRhmm, boolean smallPads, boolean pointElectrodes) {
    return solve(model, rh0mm, delRhmm, smallPads, pointElectrodes, false);
  }

  private static double[] solve(Model model, double rh0mm, double delRhmm, boolean smallPads, boolean pointElectrodes, boolean windowElectrodes) {
    model.param().set("Rh0", rh0mm + "[mm]");
    model.param().set("delRh", delRhmm + "[mm]");
    model.component("comp1").geom("geom1").run();
    if (windowElectrodes) {
      updateWindowSelections(model, rh0mm - delRhmm);
    } else if (pointElectrodes) {
      updatePointSelections(model, rh0mm - delRhmm);
    } else if (smallPads) {
      updateSmallPadSelections(model, rh0mm - delRhmm);
    } else {
      updateDomainSelections(model, rh0mm - delRhmm);
    }
    model.component("comp1").mesh("mesh1").run();
    return solveCurrentPhysics(model);
  }

  private static double[] solveCurrentPhysics(Model model) {
    model.study("std1").run();
    model.result().numerical("gevZ").set("data", "dset1");
    double[][] real = model.result().numerical("gevZ").getReal();
    System.out.println("RESULT_ROWS=" + real.length + ",COLS=" + (real.length > 0 ? real[0].length : 0));
    if (real.length < 5 || real[0].length < 1) {
      throw new RuntimeException("EvalGlobal returned an empty result for ZTT");
    }
    return new double[]{real[0][0], real[1][0], real[2][0], real[3][0], real[4][0]};
  }

  private static void setElectrodes(Model model, int sourceDomain, int groundDomain, int voltLeftDomain, int voltRightDomain) {
    model.component("comp1").physics("ec").feature("term1").selection().set(sourceDomain);
    int[] groundBoundaries = model.component("comp1").geom("geom1").getAdj(3, 2, groundDomain);
    model.component("comp1").physics("ec").feature("gnd1").selection().set(groundBoundaries);
    model.component("comp1").cpl("aveVL").selection().set(voltLeftDomain);
    model.component("comp1").cpl("aveVR").selection().set(voltRightDomain);
  }

  private static void electrodeScan(Model model) throws IOException {
    int[] e = ELECTRODE_DOMAINS;
    PrintWriter out = new PrintWriter(new FileWriter(OUT_ELECTRODE_CSV));
    try {
      out.println("source_domain,ground_domain,volt_left_domain,volt_right_domain,ZTT_ohm,VR_V,VL_V");
      out.flush();
      for (int is = 0; is < e.length; is++) {
        for (int ig = 0; ig < e.length; ig++) {
          if (ig == is) {
            continue;
          }
          for (int il = 0; il < e.length; il++) {
            if (il == is || il == ig) {
              continue;
            }
            for (int ir = 0; ir < e.length; ir++) {
              if (ir == is || ir == ig || ir == il) {
                continue;
              }
              setElectrodes(model, e[is], e[ig], e[il], e[ir]);
              double[] r = solveCurrentPhysics(model);
              out.println(e[is] + "," + e[ig] + "," + e[il] + "," + e[ir] + "," + r[0] + "," + r[1] + "," + r[2]);
              out.flush();
              System.out.println("ELECTRODE_SCAN," + e[is] + "," + e[ig] + "," + e[il] + "," + e[ir] + "," + r[0]);
            }
          }
        }
      }
    } finally {
      out.close();
    }
  }

  private static void writeBaseline(Model model) throws IOException {
    writeBaseline(model, OUT_BASELINE_CSV, false);
  }

  private static void writeBaseline(Model model, String outPath, boolean smallPads) throws IOException {
    writeBaseline(model, outPath, smallPads, false);
  }

  private static void writeBaseline(Model model, String outPath, boolean smallPads, boolean pointElectrodes) throws IOException {
    writeBaseline(model, outPath, smallPads, pointElectrodes, false);
  }

  private static void writeBaseline(Model model, String outPath, boolean smallPads, boolean pointElectrodes, boolean windowElectrodes) throws IOException {
    double[] r = solve(model, 42.5, 0.0, smallPads, pointElectrodes, windowElectrodes);
    PrintWriter out = new PrintWriter(new FileWriter(outPath));
    try {
      out.println("Rh0_mm,delRh_mm,ZTT_ohm,VR_V,VL_V,dZ_from_Z1_ohm,dZ_from_Z2_ohm");
      out.println("42.5,0," + r[0] + "," + r[1] + "," + r[2] + "," + r[3] + "," + r[4]);
      out.flush();
    } finally {
      out.close();
    }
    System.out.println((windowElectrodes ? "BASELINE_WINDOW" : (pointElectrodes ? "BASELINE_POINT" : (smallPads ? "BASELINE_SMALLPAD" : "BASELINE"))) + ",42.5,0," + Arrays.toString(r));
  }

  private static void writeCurrentBaseline(Model model, String outPath, String label) throws IOException {
    double[] r = solveCurrentPhysics(model);
    PrintWriter out = new PrintWriter(new FileWriter(outPath));
    try {
      out.println("Rh0_mm,delRh_mm,ZTT_ohm,VR_V,VL_V,dZ_from_Z1_ohm,dZ_from_Z2_ohm");
      out.println("42.5,0," + r[0] + "," + r[1] + "," + r[2] + "," + r[3] + "," + r[4]);
      out.flush();
    } finally {
      out.close();
    }
    System.out.println(label + ",42.5,0," + Arrays.toString(r));
  }

  private static double calibrateR0(Model model) throws IOException {
    double bestRh0 = 42.5;
    double bestAbs = Double.POSITIVE_INFINITY;
    PrintWriter out = new PrintWriter(new FileWriter(OUT_R0_CSV));
    try {
      out.println("Rh0_mm,delRh_mm,ZTT_ohm,VR_V,VL_V,dZ_from_Z1_ohm,dZ_from_Z2_ohm");
      out.flush();
      for (double rh0 = 40.0; rh0 <= 65.0001; rh0 += 1.0) {
        double[] r = solve(model, rh0, 0.0);
        out.println(rh0 + ",0," + r[0] + "," + r[1] + "," + r[2] + "," + r[3] + "," + r[4]);
        out.flush();
        double abs = Math.abs(r[3]);
        if (abs < bestAbs) {
          bestAbs = abs;
          bestRh0 = rh0;
        }
        System.out.println("R0_SCAN," + rh0 + "," + r[0] + "," + r[3]);
      }
    } finally {
      out.close();
    }
    System.out.println("R0_BEST," + bestRh0 + "," + bestAbs);
    return bestRh0;
  }

  private static void sweepDelRh(Model model, double rh0mm) throws IOException {
    sweepDelRh(model, rh0mm, 0.0, 10.0, 0.25, OUT_DELRH_CSV);
  }

  private static void sweepDelRhQuick(Model model, double rh0mm) throws IOException {
    sweepDelRh(model, rh0mm, 0.0, 3.0, 0.5, OUT_DELRH_QUICK_CSV);
  }

  private static void sweepDelRhQuickSmallPad(Model model, double rh0mm) throws IOException {
    sweepDelRh(model, rh0mm, 0.0, 3.0, 0.5, OUT_SMALLPAD_DELRH_QUICK_CSV, true);
  }

  private static void sweepDelRhQuickPoint(Model model, double rh0mm) throws IOException {
    sweepDelRh(model, rh0mm, 0.0, 3.0, 0.5, OUT_POINT_DELRH_QUICK_CSV, false, true);
  }

  private static void sweepDelRhQuickWindow(Model model, double rh0mm) throws IOException {
    sweepDelRh(model, rh0mm, 0.0, 3.0, 0.5, OUT_WINDOW_DELRH_QUICK_CSV, false, false, true);
  }

  private static void sweepDelRh(Model model, double rh0mm, double startMm, double endMm, double stepMm, String outPath) throws IOException {
    sweepDelRh(model, rh0mm, startMm, endMm, stepMm, outPath, false);
  }

  private static void sweepDelRh(Model model, double rh0mm, double startMm, double endMm, double stepMm, String outPath, boolean smallPads) throws IOException {
    sweepDelRh(model, rh0mm, startMm, endMm, stepMm, outPath, smallPads, false);
  }

  private static void sweepDelRh(Model model, double rh0mm, double startMm, double endMm, double stepMm, String outPath, boolean smallPads, boolean pointElectrodes) throws IOException {
    sweepDelRh(model, rh0mm, startMm, endMm, stepMm, outPath, smallPads, pointElectrodes, false);
  }

  private static void sweepDelRh(Model model, double rh0mm, double startMm, double endMm, double stepMm, String outPath, boolean smallPads, boolean pointElectrodes, boolean windowElectrodes) throws IOException {
    PrintWriter out = new PrintWriter(new FileWriter(outPath));
    try {
      out.println("Rh0_mm,delRh_mm,ZTT_ohm,VR_V,VL_V,dZ_from_Z1_ohm,dZ_from_Z2_ohm");
      out.flush();
      for (double delRh = startMm; delRh <= endMm + 0.0001; delRh += stepMm) {
        double[] r = solve(model, rh0mm, delRh, smallPads, pointElectrodes, windowElectrodes);
        out.println(rh0mm + "," + delRh + "," + r[0] + "," + r[1] + "," + r[2] + "," + r[3] + "," + r[4]);
        out.flush();
        System.out.println((windowElectrodes ? "DELRH_SCAN_WINDOW" : (pointElectrodes ? "DELRH_SCAN_POINT" : (smallPads ? "DELRH_SCAN_SMALLPAD" : "DELRH_SCAN"))) + "," + rh0mm + "," + delRh + "," + r[0] + "," + r[4]);
      }
    } finally {
      out.close();
    }
  }

  private static Model runModel() throws IOException {
    try {
      Model model = buildModel();
      writeBaseline(model);
      model.save(OUT_MPH);
      System.out.println("SAVED=" + OUT_MPH);
      return model;
    } catch (Throwable ex) {
      ex.printStackTrace();
      throw new RuntimeException(ex);
    }
  }

  private static Model runSmallPadModel(boolean quickSweep) throws IOException {
    try {
      Model model = buildModel(true);
      writeBaseline(model, OUT_SMALLPAD_BASELINE_CSV, true);
      if (quickSweep) {
        sweepDelRhQuickSmallPad(model, 42.5);
      }
      model.save(OUT_SMALLPAD_MPH);
      System.out.println("SAVED=" + OUT_SMALLPAD_MPH);
      return model;
    } catch (Throwable ex) {
      ex.printStackTrace();
      throw new RuntimeException(ex);
    }
  }

  private static Model runPointModel(boolean quickSweep) throws IOException {
    try {
      Model model = buildModel(false, true);
      writeBaseline(model, OUT_POINT_BASELINE_CSV, false, true);
      if (quickSweep) {
        sweepDelRhQuickPoint(model, 42.5);
      }
      model.save(OUT_POINT_MPH);
      System.out.println("SAVED=" + OUT_POINT_MPH);
      return model;
    } catch (Throwable ex) {
      ex.printStackTrace();
      throw new RuntimeException(ex);
    }
  }

  private static Model runWindowModel(boolean quickSweep) throws IOException {
    try {
      Model model = buildModel(false, false, true);
      writeBaseline(model, OUT_WINDOW_BASELINE_CSV, false, false, true);
      if (quickSweep) {
        sweepDelRhQuickWindow(model, 42.5);
      }
      model.save(OUT_WINDOW_MPH);
      System.out.println("SAVED=" + OUT_WINDOW_MPH);
      return model;
    } catch (Throwable ex) {
      ex.printStackTrace();
      throw new RuntimeException(ex);
    }
  }

  private static Model runSkinGaussianModel() throws IOException {
    try {
      Model model = buildModel(false, false, false, true);
      writeCurrentBaseline(model, OUT_SKIN_GAUSSIAN_BASELINE_CSV, "BASELINE_SKIN_GAUSSIAN");
      model.save(OUT_SKIN_GAUSSIAN_MPH);
      System.out.println("SAVED=" + OUT_SKIN_GAUSSIAN_MPH);
      return model;
    } catch (Throwable ex) {
      ex.printStackTrace();
      throw new RuntimeException(ex);
    }
  }

  public static void main(String[] args) throws IOException {
    try {
      mainImpl(args);
    } catch (Throwable ex) {
      ex.printStackTrace();
      throw new RuntimeException(ex);
    }
  }

  private static void mainImpl(String[] args) throws IOException {
    if (args.length > 0 && "smallpads".equalsIgnoreCase(args[0])) {
      runSmallPadModel(false);
      return;
    }
    if (args.length > 0 && "smallpads_delrhquick".equalsIgnoreCase(args[0])) {
      runSmallPadModel(true);
      return;
    }
    if (args.length > 0 && "point".equalsIgnoreCase(args[0])) {
      runPointModel(false);
      return;
    }
    if (args.length > 0 && "point_homogeneous".equalsIgnoreCase(args[0])) {
      Model model = buildModel(false, true);
      makeHomogeneousSoft(model);
      writeBaseline(model, OUT_POINT_HOM_BASELINE_CSV, false, true);
      return;
    }
    if (args.length > 0 && "point_delrhquick".equalsIgnoreCase(args[0])) {
      runPointModel(true);
      return;
    }
    if (args.length > 0 && "window".equalsIgnoreCase(args[0])) {
      runWindowModel(false);
      return;
    }
    if (args.length > 0 && "skin_gaussian".equalsIgnoreCase(args[0])) {
      runSkinGaussianModel();
      return;
    }
    if (args.length > 0 && "window_homogeneous".equalsIgnoreCase(args[0])) {
      Model model = buildModel(false, false, true);
      makeHomogeneousSoft(model);
      writeBaseline(model, OUT_WINDOW_HOM_BASELINE_CSV, false, false, true);
      return;
    }
    if (args.length > 0 && "window_rho_div10".equalsIgnoreCase(args[0])) {
      Model model = buildModel(false, false, true);
      divideOwnRhosBy10(model);
      writeBaseline(model, OUT_WINDOW_RHO10_BASELINE_CSV, false, false, true);
      return;
    }
    if (args.length > 0 && "window_rho_match_z1".equalsIgnoreCase(args[0])) {
      Model model = buildModel(false, false, true);
      setOwnRhosForZ1(model);
      writeBaseline(model, OUT_WINDOW_RHO_MATCH_BASELINE_CSV, false, false, true);
      return;
    }
    if (args.length > 0 && "window_soft_match_lung_ct".equalsIgnoreCase(args[0])) {
      Model model = buildModel(false, false, true);
      setSoftForZ1KeepCtLung(model);
      writeBaseline(model, OUT_WINDOW_SOFT_MATCH_LUNG_CT_BASELINE_CSV, false, false, true);
      return;
    }
    if (args.length > 0 && "window_soft031_lung_ct".equalsIgnoreCase(args[0])) {
      Model model = buildModel(false, false, true);
      setSoft031KeepCtLung(model);
      writeBaseline(model, OUT_WINDOW_SOFT031_LUNG_CT_BASELINE_CSV, false, false, true);
      return;
    }
    if (args.length > 0 && "window_delrhquick".equalsIgnoreCase(args[0])) {
      runWindowModel(true);
      return;
    }
    if (args.length > 0 && "maponly".equalsIgnoreCase(args[0])) {
      Model model = buildModel();
      double rh0 = args.length > 1 ? Double.parseDouble(args[1]) : 42.5;
      mapDelRh(model, rh0);
      model.save(OUT_MPH);
      return;
    }
    Model model = runModel();
    if (args.length > 0 && "full".equalsIgnoreCase(args[0])) {
      double bestRh0 = calibrateR0(model);
      sweepDelRh(model, bestRh0);
      model.save(OUT_MPH);
    } else if (args.length > 0 && "electrodes".equalsIgnoreCase(args[0])) {
      electrodeScan(model);
      model.save(OUT_MPH);
    } else if (args.length > 0 && "r0only".equalsIgnoreCase(args[0])) {
      calibrateR0(model);
      model.save(OUT_MPH);
    } else if (args.length > 0 && "delrhquick".equalsIgnoreCase(args[0])) {
      double rh0 = args.length > 1 ? Double.parseDouble(args[1]) : 42.5;
      sweepDelRhQuick(model, rh0);
      model.save(OUT_MPH);
    } else if (args.length > 0 && "delrhonly".equalsIgnoreCase(args[0])) {
      double rh0 = args.length > 1 ? Double.parseDouble(args[1]) : 42.5;
      sweepDelRh(model, rh0);
      model.save(OUT_MPH);
    }
  }
}
