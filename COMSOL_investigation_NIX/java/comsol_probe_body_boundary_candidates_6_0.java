import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class comsol_probe_body_boundary_candidates_6_0 {
  private static final String STEP_PATH = "C:\\tmp\\TTrans_for_NIX.step";
  private static final double HEART_CX = -0.026;
  private static final double HEART_CY = 0.0355;
  private static final double HEART_CZ = 0.026;
  private static final double[] X_TARGETS = new double[]{-0.295, -0.197, 0.197, 0.295};

  private static class Face {
    int id;
    double area;
    double[] b;
    Face(int id, double area, double[] b) {
      this.id = id;
      this.area = area;
      this.b = b;
    }
    double cx() { return 0.5 * (b[0] + b[1]); }
    double cy() { return 0.5 * (b[2] + b[3]); }
    double cz() { return 0.5 * (b[4] + b[5]); }
    double dx() { return b[1] - b[0]; }
    double dy() { return b[3] - b[2]; }
    double dz() { return b[5] - b[4]; }
    double intervalDistance(double value, double lo, double hi) {
      if (value < lo) return lo - value;
      if (value > hi) return value - hi;
      return 0.0;
    }
    double xPenalty(double x) { return intervalDistance(x, b[0], b[1]); }
    String line(String tag, double x) {
      return tag + ",targetX=" + x + ",boundary=" + id + ",area=" + area +
        ",center=[" + cx() + "," + cy() + "," + cz() + "],span=[" + dx() + "," + dy() + "," + dz() + "],bbox=" + Arrays.toString(b);
    }
  }

  public static void main(String[] args) {
    Model model = ModelUtil.create("BodyBoundaryCandidates");
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
      if (volume > bestVolume) {
        soft = domain;
        bestVolume = volume;
      }
    }
    System.out.println("SOFT_DOMAIN=" + soft + ",volume=" + bestVolume);
    int[] boundaries = model.component("comp1").geom("geom1").getAdj(3, 2, soft);
    List<Face> faces = new ArrayList<Face>();
    for (int i = 0; i < boundaries.length; i++) {
      int b = boundaries[i];
      model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 2);
      model.component("comp1").geom("geom1").measureFinal().selection().set(b);
      faces.add(new Face(b, model.component("comp1").geom("geom1").measureFinal().getVolume(),
        model.component("comp1").geom("geom1").measureFinal().getBoundingBox()));
    }

    for (int t = 0; t < X_TARGETS.length; t++) {
      final double x = X_TARGETS[t];
      List<Face> near = new ArrayList<Face>();
      for (int i = 0; i < faces.size(); i++) {
        Face f = faces.get(i);
        if (f.xPenalty(x) <= 0.01) near.add(f);
      }
      near.sort(new Comparator<Face>() {
        public int compare(Face a, Face b) {
          return Double.compare(b.cz(), a.cz());
        }
      });
      for (int i = 0; i < Math.min(8, near.size()); i++) {
        System.out.println(near.get(i).line("TOP_BY_Z_RANK" + (i + 1), x));
      }
      near.sort(new Comparator<Face>() {
        public int compare(Face a, Face b) {
          return Double.compare(b.cy(), a.cy());
        }
      });
      for (int i = 0; i < Math.min(8, near.size()); i++) {
        System.out.println(near.get(i).line("TOP_BY_Y_RANK" + (i + 1), x));
      }
    }
  }
}
