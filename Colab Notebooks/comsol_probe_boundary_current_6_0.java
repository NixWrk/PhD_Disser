import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.Arrays;

public class comsol_probe_boundary_current_6_0 {
  public static void main(String[] args) {
    Model model = ModelUtil.create("BoundaryCurrentProbe");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").geom("geom1").create("blk1", "Block");
    model.component("comp1").geom("geom1").feature("blk1").set("size", new double[]{1, 1, 1});
    model.component("comp1").geom("geom1").run();
    model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
    model.component("comp1").physics("ec").create("bcs1", "BoundaryCurrentSource", 2);
    System.out.println("BCS_PROPS=" + Arrays.toString(model.component("comp1").physics("ec").feature("bcs1").properties()));
    model.component("comp1").physics("ec").create("ncd1", "NormalCurrentDensity", 2);
    System.out.println("NCD_PROPS=" + Arrays.toString(model.component("comp1").physics("ec").feature("ncd1").properties()));
  }
}
