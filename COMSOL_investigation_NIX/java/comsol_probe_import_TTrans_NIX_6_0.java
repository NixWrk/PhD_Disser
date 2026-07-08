import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.Arrays;

public class comsol_probe_import_TTrans_NIX_6_0 {
  private static final String STEP_PATH = "C:\\tmp\\TTrans_for_NIX.step";
  private static final String OUT_PATH = "C:\\tmp\\TTrans_for_NIX_import_probe_6_0.mph";

  public static Model run() throws java.io.IOException {
    Model model = ModelUtil.create("TTransProbe");
    model.modelPath("C:\\tmp");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").geom("geom1").create("imp1", "Import");
    model.component("comp1").geom("geom1").feature("imp1").set("filename", STEP_PATH);
    model.component("comp1").geom("geom1").feature("imp1").set("unit", "source");
    model.component("comp1").geom("geom1").feature("imp1").importData();
    model.component("comp1").geom("geom1").run();

    System.out.println("OBJECT_NAMES=" + Arrays.toString(model.component("comp1").geom("geom1").objectNames()));
    System.out.println("N_ENTITIES=" + Arrays.toString(model.component("comp1").geom("geom1").getNEntities()));
    System.out.println("BOUNDING_BOX=" + Arrays.toString(model.component("comp1").geom("geom1").getBoundingBox()));

    String[] objectNames = model.component("comp1").geom("geom1").objectNames();
    for (int i = 0; i < objectNames.length; i++) {
      String objectName = objectNames[i];
      try {
        model.component("comp1").geom("geom1").measure().selection().init(3);
        model.component("comp1").geom("geom1").measure().selection().all(objectName);
        System.out.println(
          "OBJECT_MAP," + objectName +
          ",volume," + model.component("comp1").geom("geom1").measure().getVolume() +
          ",bbox," + Arrays.toString(model.component("comp1").geom("geom1").measure().getBoundingBox())
        );
      } catch (Exception ex) {
        System.out.println("OBJECT_MAP_ERROR," + objectName + "," + ex.getMessage());
      }
    }

    int[] nEntities = model.component("comp1").geom("geom1").getNEntities();
    int domainCount = nEntities.length > 3 ? nEntities[3] : 0;
    for (int domain = 1; domain <= domainCount; domain++) {
      try {
        model.component("comp1").geom("geom1").measureFinal().selection().geom("geom1", 3);
        model.component("comp1").geom("geom1").measureFinal().selection().set(domain);
        System.out.println(
          "DOMAIN_MAP," + domain +
          ",volume," + model.component("comp1").geom("geom1").measureFinal().getVolume() +
          ",bbox," + Arrays.toString(model.component("comp1").geom("geom1").measureFinal().getBoundingBox())
        );
      } catch (Exception ex) {
        System.out.println("DOMAIN_MAP_ERROR," + domain + "," + ex.getMessage());
      }
    }

    model.save(OUT_PATH);
    System.out.println("SAVED=" + OUT_PATH);
    return model;
  }

  public static void main(String[] args) throws java.io.IOException {
    run();
  }
}
