import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.File;
import java.util.Arrays;

public class comsol_export_visualization_6_0 {
  private static final String POINT_MPH = "C:\\tmp\\TTrans_NIX_point_radius_6_0.mph";
  private static final String WINDOW_MPH = "C:\\tmp\\TTrans_NIX_window_radius_6_0.mph";
  private static final String OUT_DIR = "C:\\tmp\\comsol_visualization";

  private static void ensureOutDir() {
    File dir = new File(OUT_DIR);
    if (!dir.exists()) {
      dir.mkdirs();
    }
  }

  private static void exportGeometry(Model model, String prefix) {
    model.component("comp1").geom("geom1").image().set("pngfilename", OUT_DIR + "\\" + prefix + "_geometry.png");
    model.component("comp1").geom("geom1").image().set("width", "1600");
    model.component("comp1").geom("geom1").image().set("height", "1000");
    model.component("comp1").geom("geom1").image().set("unit", "px");
    model.component("comp1").geom("geom1").image().export();
    System.out.println("EXPORTED=" + OUT_DIR + "\\" + prefix + "_geometry.png");
  }

  private static void exportMaterial(Model model, String matTag, String fileName) {
    model.component("comp1").material(matTag).image().set("pngfilename", OUT_DIR + "\\" + fileName);
    model.component("comp1").material(matTag).image().set("width", "1600");
    model.component("comp1").material(matTag).image().set("height", "1000");
    model.component("comp1").material(matTag).image().set("unit", "px");
    model.component("comp1").material(matTag).image().export();
    System.out.println("EXPORTED=" + OUT_DIR + "\\" + fileName);
  }

  private static void exportPhysics(Model model, String prefix) {
    model.component("comp1").physics("ec").image().set("pngfilename", OUT_DIR + "\\" + prefix + "_ec_physics.png");
    model.component("comp1").physics("ec").image().set("width", "1600");
    model.component("comp1").physics("ec").image().set("height", "1000");
    model.component("comp1").physics("ec").image().set("unit", "px");
    model.component("comp1").physics("ec").image().export();
    System.out.println("EXPORTED=" + OUT_DIR + "\\" + prefix + "_ec_physics.png");
  }

  private static void exportModel(String modelPath, String prefix) throws Exception {
    Model model = ModelUtil.load(prefix, modelPath);
    System.out.println("LOADED=" + modelPath);
    System.out.println("RESULT_TAGS=" + Arrays.toString(model.result().tags()));

    exportGeometry(model, prefix);
    exportMaterial(model, "matSoft", prefix + "_mat_soft.png");
    exportMaterial(model, "matLung", prefix + "_mat_lung.png");
    exportMaterial(model, "matMyocardium", prefix + "_mat_myocardium.png");
    exportMaterial(model, "matBlood", prefix + "_mat_blood.png");
    exportMaterial(model, "matElectrode", prefix + "_mat_electrode.png");
    exportPhysics(model, prefix);
  }

  public static void main(String[] args) throws Exception {
    ensureOutDir();
    exportModel(POINT_MPH, "point");
    exportModel(WINDOW_MPH, "window");
  }
}
