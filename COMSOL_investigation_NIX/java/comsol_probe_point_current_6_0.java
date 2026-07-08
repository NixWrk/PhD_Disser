import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.Arrays;

public class comsol_probe_point_current_6_0 {
  private static double cx(double[] bbox) {
    return 0.5 * (bbox[0] + bbox[1]);
  }

  private static double cy(double[] bbox) {
    return 0.5 * (bbox[2] + bbox[3]);
  }

  private static double cz(double[] bbox) {
    return 0.5 * (bbox[4] + bbox[5]);
  }

  private static int closestPoint(Model model, double x, double y, double z) {
    int[] n = model.component("comp1").geom("geom1").getNEntities();
    int points = n.length > 0 ? n[0] : 0;
    int best = -1;
    double bestDist = Double.POSITIVE_INFINITY;
    for (int p = 1; p <= points; p++) {
      model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 0);
      model.component("comp1").geom("geom1").measureFinal().selection().set(p);
      double[] bbox = model.component("comp1").geom("geom1").measureFinal().getBoundingBox();
      double dx = cx(bbox) - x;
      double dy = cy(bbox) - y;
      double dz = cz(bbox) - z;
      double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      System.out.println("POINT,id=" + p + ",bbox=" + Arrays.toString(bbox) + ",dist=" + dist);
      if (dist < bestDist) {
        best = p;
        bestDist = dist;
      }
    }
    return best;
  }

  public static void main(String[] args) throws IOException {
    Model model = ModelUtil.create("PointProbe");
    model.modelPath("C:\\tmp");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").geom("geom1").create("blk1", "Block");
    model.component("comp1").geom("geom1").feature("blk1").set("size", new double[]{1, 1, 1});
    model.component("comp1").geom("geom1").feature("blk1").set("base", "center");

    model.component("comp1").geom("geom1").create("ptA", "Point");
    model.component("comp1").geom("geom1").feature("ptA").set("p", new double[]{-0.3, 0.0, 0.0});
    model.component("comp1").geom("geom1").create("ptB", "Point");
    model.component("comp1").geom("geom1").feature("ptB").set("p", new double[]{0.3, 0.0, 0.0});
    model.component("comp1").geom("geom1").create("ptM", "Point");
    model.component("comp1").geom("geom1").feature("ptM").set("p", new double[]{-0.1, 0.0, 0.0});
    model.component("comp1").geom("geom1").create("ptN", "Point");
    model.component("comp1").geom("geom1").feature("ptN").set("p", new double[]{0.1, 0.0, 0.0});
    model.component("comp1").geom("geom1").run();
    System.out.println("NENT=" + Arrays.toString(model.component("comp1").geom("geom1").getNEntities()));

    int pA = closestPoint(model, -0.3, 0.0, 0.0);
    int pB = closestPoint(model, 0.3, 0.0, 0.0);
    int pM = closestPoint(model, -0.1, 0.0, 0.0);
    int pN = closestPoint(model, 0.1, 0.0, 0.0);
    System.out.println("PICKS=" + Arrays.toString(new int[]{pA, pB, pM, pN}));

    model.component("comp1").material().create("mat1", "Common");
    model.component("comp1").material("mat1").selection().set(1);
    model.component("comp1").material("mat1").propertyGroup("def").set("electricconductivity", new String[]{"1[S/m]"});
    model.component("comp1").material("mat1").propertyGroup("def").set("relpermittivity", new String[]{"1"});

    model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
    model.component("comp1").physics("ec").create("pcsA", "PointCurrentSource", 0);
    System.out.println("PCS_PROPS=" + Arrays.toString(model.component("comp1").physics("ec").feature("pcsA").properties()));
    model.component("comp1").physics("ec").feature("pcsA").selection().set(pA);
    model.component("comp1").physics("ec").feature("pcsA").set("Qjp", "1[A]");
    model.component("comp1").physics("ec").create("pcsB", "PointCurrentSource", 0);
    model.component("comp1").physics("ec").feature("pcsB").selection().set(pB);
    model.component("comp1").physics("ec").feature("pcsB").set("Qjp", "-1[A]");

    model.component("comp1").cpl().create("aveM", "Average");
    model.component("comp1").cpl("aveM").selection().geom("geom1", 0);
    model.component("comp1").cpl("aveM").selection().set(pM);
    model.component("comp1").cpl().create("aveN", "Average");
    model.component("comp1").cpl("aveN").selection().geom("geom1", 0);
    model.component("comp1").cpl("aveN").selection().set(pN);
    model.component("comp1").variable().create("var1");
    model.component("comp1").variable("var1").set("Z", "abs(aveN(V)-aveM(V))/1[A]");

    model.component("comp1").mesh("mesh1").autoMeshSize(5);
    model.component("comp1").mesh("mesh1").run();
    model.study().create("std1");
    model.study("std1").create("stat", "Stationary");
    model.study("std1").run();
    model.result().numerical().create("gev", "EvalGlobal");
    model.result().numerical("gev").set("expr", new String[]{"Z", "aveM(V)", "aveN(V)"});
    model.result().numerical("gev").set("unit", new String[]{"ohm", "V", "V"});
    double[][] real = model.result().numerical("gev").getReal();
    System.out.println("RESULT=" + real[0][0] + "," + real[1][0] + "," + real[2][0]);
    model.save("C:\\tmp\\point_probe.mph");
  }
}
