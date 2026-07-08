import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.Arrays;

public class comsol_probe_normal_current_6_0 {
  private static double bx(double[] bbox) {
    return 0.5 * (bbox[0] + bbox[1]);
  }

  private static int faceByX(Model model, double targetX) {
    int[] n = model.component("comp1").geom("geom1").getNEntities();
    int boundaries = n.length > 2 ? n[2] : 0;
    int best = -1;
    double bestDist = Double.POSITIVE_INFINITY;
    for (int b = 1; b <= boundaries; b++) {
      model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 2);
      model.component("comp1").geom("geom1").measureFinal().selection().set(b);
      double[] bbox = model.component("comp1").geom("geom1").measureFinal().getBoundingBox();
      double dist = Math.abs(bx(bbox) - targetX) + 100.0 * (bbox[1] - bbox[0]);
      System.out.println("BND,id=" + b + ",bbox=" + Arrays.toString(bbox) + ",score=" + dist);
      if (dist < bestDist) {
        best = b;
        bestDist = dist;
      }
    }
    return best;
  }

  public static void main(String[] args) {
    Model model = ModelUtil.create("NormalCurrentProbe");
    model.modelPath("C:\\tmp");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").geom("geom1").create("blk1", "Block");
    model.component("comp1").geom("geom1").feature("blk1").set("size", new double[]{1, 1, 1});
    model.component("comp1").geom("geom1").feature("blk1").set("base", "center");
    model.component("comp1").geom("geom1").run();
    int left = faceByX(model, -0.5);
    int right = faceByX(model, 0.5);
    System.out.println("PICKS=" + left + "," + right);

    model.component("comp1").material().create("mat1", "Common");
    model.component("comp1").material("mat1").selection().set(1);
    model.component("comp1").material("mat1").propertyGroup("def").set("electricconductivity", new String[]{"1[S/m]"});
    model.component("comp1").material("mat1").propertyGroup("def").set("relpermittivity", new String[]{"1"});
    model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
    model.component("comp1").physics("ec").create("ncdL", "NormalCurrentDensity", 2);
    model.component("comp1").physics("ec").feature("ncdL").selection().set(left);
    model.component("comp1").physics("ec").feature("ncdL").set("nJ", "-1[A/m^2]");
    model.component("comp1").physics("ec").create("ncdR", "NormalCurrentDensity", 2);
    model.component("comp1").physics("ec").feature("ncdR").selection().set(right);
    model.component("comp1").physics("ec").feature("ncdR").set("nJ", "1[A/m^2]");
    model.component("comp1").physics("ec").create("ref1", "ElectricPotential", 0);
    model.component("comp1").physics("ec").feature("ref1").selection().set(1);
    model.component("comp1").physics("ec").feature("ref1").set("V0", "0[V]");

    model.component("comp1").cpl().create("aveL", "Average");
    model.component("comp1").cpl("aveL").selection().geom("geom1", 2);
    model.component("comp1").cpl("aveL").selection().set(left);
    model.component("comp1").cpl().create("aveR", "Average");
    model.component("comp1").cpl("aveR").selection().geom("geom1", 2);
    model.component("comp1").cpl("aveR").selection().set(right);
    model.component("comp1").variable().create("var1");
    model.component("comp1").variable("var1").set("Z", "aveR(V)-aveL(V)");
    model.component("comp1").mesh("mesh1").autoMeshSize(5);
    model.component("comp1").mesh("mesh1").run();
    model.study().create("std1");
    model.study("std1").create("stat", "Stationary");
    model.study("std1").run();
    model.result().numerical().create("gev", "EvalGlobal");
    model.result().numerical("gev").set("expr", new String[]{"Z", "aveL(V)", "aveR(V)"});
    double[][] real = model.result().numerical("gev").getReal();
    System.out.println("RESULT=" + real[0][0] + "," + real[1][0] + "," + real[2][0]);
  }
}
