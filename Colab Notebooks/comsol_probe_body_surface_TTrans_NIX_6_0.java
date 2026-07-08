import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.Arrays;

public class comsol_probe_body_surface_TTrans_NIX_6_0 {
  private static final String STEP_PATH = "C:\\tmp\\TTrans_for_NIX.step";
  private static final double HEART_CX = -0.026;
  private static final double HEART_CY = 0.0355;
  private static final double HEART_CZ = 0.026;
  private static final double ELECTRODE_Y = -0.050;
  private static final double[] X_TARGETS = new double[]{-0.295, -0.197, 0.197, 0.295};

  private static double center(double lo, double hi) {
    return 0.5 * (lo + hi);
  }

  private static boolean contains(double value, double lo, double hi, double eps) {
    return value >= lo - eps && value <= hi + eps;
  }

  public static void main(String[] args) {
    Model model = ModelUtil.create("BodySurfaceProbe");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 3);

    model.param().set("Rh_geom", "42.5[mm]");
    model.param().set("Rh0", "42.5[mm]");
    model.param().set("delRh", "0[mm]");
    model.param().set("heartRadius", "Rh0-delRh");
    model.param().set("heartScale", "(Rh0-delRh)/Rh_geom");
    model.param().set("fillOuterRadius", "max(Rh_geom,heartRadius)");

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
    model.component("comp1").geom("geom1").run();

    int[] nEntities = model.component("comp1").geom("geom1").getNEntities();
    int domainCount = nEntities.length > 3 ? nEntities[3] : 0;
    int soft = -1;
    double bestVolume = -1.0;
    for (int domain = 1; domain <= domainCount; domain++) {
      model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 3);
      model.component("comp1").geom("geom1").measureFinal().selection().set(domain);
      double volume = model.component("comp1").geom("geom1").measureFinal().getVolume();
      double[] bbox = model.component("comp1").geom("geom1").measureFinal().getBoundingBox();
      System.out.println("DOMAIN,id=" + domain + ",vol=" + volume + ",bbox=" + Arrays.toString(bbox));
      if (volume > bestVolume) {
        soft = domain;
        bestVolume = volume;
      }
    }
    System.out.println("SOFT_CANDIDATE=" + soft + ",vol=" + bestVolume);

    int[] boundaries = model.component("comp1").geom("geom1").getAdj(3, 2, soft);
    for (int i = 0; i < boundaries.length; i++) {
      int b = boundaries[i];
      model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 2);
      model.component("comp1").geom("geom1").measureFinal().selection().set(b);
      double[] bbox = model.component("comp1").geom("geom1").measureFinal().getBoundingBox();
      double area = model.component("comp1").geom("geom1").measureFinal().getVolume();
      for (int j = 0; j < X_TARGETS.length; j++) {
        double x = X_TARGETS[j];
        if (contains(x, bbox[0], bbox[1], 0.004) && contains(ELECTRODE_Y, bbox[2], bbox[3], 0.004)) {
          System.out.println(
            "BODY_FACE,targetX=" + x +
            ",boundary=" + b +
            ",area=" + area +
            ",bbox=" + Arrays.toString(bbox) +
            ",center=[" + center(bbox[0], bbox[1]) + "," + center(bbox[2], bbox[3]) + "," + center(bbox[4], bbox[5]) + "]"
          );
        }
      }
    }
  }
}
