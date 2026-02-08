package dnnsg;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class DNNSG {

    // Array of skip percentages to test
    static double[] SKIP_PERCENTAGES_TO_TEST = {0.0, 0.1, 0.25, 0.5, 0.75, 1.00};
    
    // Current skip percentage being tested (set during execution)
    static double CURRENT_SKIP_PERCENTAGE = 0.0;

    // Store baseline parameter counts for each input/output class
    // Key format: "input_output" (e.g., "21_10")
    static Map<String, Integer> CLASS_BASELINE_PARAMS = new HashMap<>();
    
    // Track which student config is the baseline for each class
    // Key format: "input_output", Value: the config array string
    static Map<String, String> BASELINE_STUDENT_CONFIG = new HashMap<>();
    
    // Maximum allowed deviation from baseline (5%)
    static final double MAX_PARAM_DEVIATION = 0.05;

    // PHASE 1: Start with minimal, balanced configuration
    static int[] inputsize = {21, 15, 9};
    static int[] outputsize = {10, 8, 4};
    static int NUM_REPLICATIONS = 3;

    // FIXED: Added deep teachers + balanced parameter counts (~15K params each)
    static int[][] teacherConfigs = {
        {21, 270, 10}, {21, 130, 85, 10}, {21, 80, 80, 80, 10},
        {21, 270, 4}, {21, 130, 85, 4}, {21, 80, 80, 80, 4},
        {15, 320, 10}, {15, 150, 100, 10}, {15, 90, 90, 90, 10},
        {15, 320, 4}, {15, 150, 100, 4}, {15, 90, 90, 90, 4},
        {9, 410, 10}, {9, 200, 130, 10}, {9, 120, 120, 120, 10},
        {9, 410, 4}, {9, 200, 130, 4}, {9, 120, 120, 120, 4},
        {9, 410, 8}, {9, 200, 130, 8}, {9, 120, 120, 120, 8},
        {15, 320, 8}, {15, 150, 100, 8}, {15, 90, 90, 90, 8} 
    };

    // Enhanced student configurations (all ~700 params)
    // FIRST student in each input/output group will be the baseline
static int[][] studentConfigs = {
    // ========== 21 inputs, 10 outputs ==========
    // Shallow-Medium (4-8 layers) - Baseline for comparison
    {21, 16, 16, 10},           // 3 layers - too shallow
    {21, 12, 12, 12, 10},       // 4 layers - starting point
    {21, 10, 10, 10, 10, 10},   // 5 layers
    {21, 8, 8, 8, 8, 8, 10},    // 6 layers
    {21, 6, 6, 6, 6, 6, 6, 10}, // 7 layers
    {21, 5, 5, 5, 5, 5, 5, 5, 10}, // 8 layers
    
    // Deep (12-20 layers) - Where skips should help significantly
    {21, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 10},  // 12 layers
    {21, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 10},  // 15 layers
    {21, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10},  // 18 layers
    
    // Very Deep (30+ layers) - ResNet territory
    {21, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10},  // 32 layers
    
    // ========== 21 inputs, 4 outputs ==========
    {21, 16, 16, 4},
    {21, 12, 12, 12, 4},
    {21, 10, 10, 10, 10, 4},
    {21, 8, 8, 8, 8, 8, 4},
    {21, 6, 6, 6, 6, 6, 6, 4},
    {21, 5, 5, 5, 5, 5, 5, 5, 4},
    {21, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    {21, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4},
    {21, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4},
    
    // ========== 15 inputs, 10 outputs ==========
    {15, 18, 18, 10},
    {15, 14, 14, 14, 10},
    {15, 12, 12, 12, 12, 10},
    {15, 10, 10, 10, 10, 10, 10},
    {15, 8, 8, 8, 8, 8, 8, 8, 10},
    {15, 6, 6, 6, 6, 6, 6, 6, 6, 10},
    {15, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 10},
    {15, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 10},
    {15, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10},
    
    // ========== 15 inputs, 4 outputs ==========
    {15, 18, 18, 4},
    {15, 14, 14, 14, 4},
    {15, 12, 12, 12, 12, 4},
    {15, 10, 10, 10, 10, 10, 4},
    {15, 8, 8, 8, 8, 8, 8, 8, 4},
    {15, 6, 6, 6, 6, 6, 6, 6, 6, 4},
    {15, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    {15, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4},
    {15, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4},
    
    // ========== 15 inputs, 8 outputs ==========
    {15, 18, 18, 8},
    {15, 14, 14, 14, 8},
    {15, 12, 12, 12, 12, 8},
    {15, 10, 10, 10, 10, 10, 8},
    {15, 8, 8, 8, 8, 8, 8, 8, 8},
    {15, 6, 6, 6, 6, 6, 6, 6, 6, 8},
    {15, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8},
    {15, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8},
    {15, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8},
    
    // ========== 9 inputs, 10 outputs ==========
    {9, 20, 20, 10},
    {9, 16, 16, 16, 10},
    {9, 14, 14, 14, 14, 10},
    {9, 12, 12, 12, 12, 12, 10},
    {9, 10, 10, 10, 10, 10, 10, 10},
    {9, 8, 8, 8, 8, 8, 8, 8, 8, 10},
    {9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 10},
    {9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 10},
    {9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 10},
    
    // ========== 9 inputs, 4 outputs ==========
    {9, 20, 20, 4},
    {9, 16, 16, 16, 4},
    {9, 14, 14, 14, 14, 4},
    {9, 12, 12, 12, 12, 12, 4},
    {9, 10, 10, 10, 10, 10, 10, 4},
    {9, 8, 8, 8, 8, 8, 8, 8, 8, 4},
    {9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4},
    {9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    {9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4},
    
    // ========== 9 inputs, 8 outputs ==========
    {9, 20, 20, 8},
    {9, 16, 16, 16, 8},
    {9, 14, 14, 14, 14, 8},
    {9, 12, 12, 12, 12, 12, 8},
    {9, 10, 10, 10, 10, 10, 10, 8},
    {9, 8, 8, 8, 8, 8, 8, 8, 8, 8},
    {9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8},
    {9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8},
    {9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8}
};
    
    static int numdatapoints = 10000;
    static int numValdatapoints = 1000;
    static int numepochs = 200;

    static String mainTeacherFolder = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\TeacherArchitectures60.1\\";

    private static void ensureDirectoryExists(String filePath) {
        File file = new File(filePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
    }

    // Helper to create unique key for each class (input_output)
    private static String getClassKey(int input, int output) {
        return input + "_" + output;
    }

    // Helper to create config string for tracking
    private static String configToString(int[] config) {
        return java.util.Arrays.toString(config);
    }

    // NEW: Calculate and store baseline parameters by selecting first student in each class
    public static void calculateBaselineParameters() {
        CURRENT_SKIP_PERCENTAGE = 0.0;
        
        System.out.println("\n" + "â•".repeat(70));
        System.out.println("PHASE 1: CALCULATING STUDENT BASELINES BY CLASS (0% SKIP)");
        System.out.println("â•".repeat(70) + "\n");
        
        String baselineLogPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\ExperimentSet_SkipAnalysis60.1\\baseline_params.txt";
        ensureDirectoryExists(baselineLogPath);
        
        try (PrintWriter baselineWriter = new PrintWriter(new FileWriter(baselineLogPath))) {
            baselineWriter.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
            baselineWriter.println("STUDENT BASELINE PARAMETERS BY CLASS (0% Skip Connections)");
            baselineWriter.println("First student in each (input, output) class serves as baseline");
            baselineWriter.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
            baselineWriter.println();
            
            for (int input : inputsize) {
                for (int output : outputsize) {
                    List<int[]> matchingStudents = getMatchingStudents(input, output);
                    
                    if (matchingStudents.isEmpty()) continue;
                    
                    String classKey = getClassKey(input, output);
                    
                    System.out.println("\nInput=" + input + ", Output=" + output + ":");
                    baselineWriter.println("Input=" + input + ", Output=" + output + ":");
                    
                    // FIRST student in this class is the baseline
                    int[] baselineConfig = matchingStudents.get(0);
                    int baselineParams = createAndCountParams(baselineConfig, 0.0);
                    
                    CLASS_BASELINE_PARAMS.put(classKey, baselineParams);
                    BASELINE_STUDENT_CONFIG.put(classKey, configToString(baselineConfig));
                    
                    String archStr = architectureToString(baselineConfig);
                    String depthStr = getStudentDepth(baselineConfig);
                    
                    System.out.println("  âœ“ BASELINE: " + archStr + " (" + depthStr + "): " + 
                                     baselineParams + " params");
                    
                    baselineWriter.println("  âœ“ BASELINE: " + archStr + " (" + depthStr + "): " + 
                                         baselineParams + " params");
                    baselineWriter.println("    (All other students in this class will match this count)");
                    baselineWriter.println();
                    
                    // Show other students that need to match this baseline
                    for (int i = 1; i < matchingStudents.size(); i++) {
                        int[] otherConfig = matchingStudents.get(i);
                        String otherArch = architectureToString(otherConfig);
                        String otherDepth = getStudentDepth(otherConfig);
                        
                        System.out.println("  â†’ Other: " + otherArch + " (" + otherDepth + ") will be adjusted to " + 
                                         baselineParams + " params");
                        
                        baselineWriter.println("  â†’ Other: " + otherArch + " (" + otherDepth + ") will be adjusted to " + 
                                             baselineParams + " params");
                    }
                    baselineWriter.println();
                }
            }
            
            baselineWriter.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
            baselineWriter.println("Total Classes: " + CLASS_BASELINE_PARAMS.size());
            baselineWriter.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
            
        } catch (IOException e) {
            System.out.println("Error writing baseline parameters file: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n" + "â•".repeat(70));
        System.out.println("âœ… STUDENT BASELINES CALCULATED: " + CLASS_BASELINE_PARAMS.size() + " classes");
        System.out.println("â•".repeat(70) + "\n");
    }

    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        try {
            ensureDirectoryExists(mainTeacherFolder + "dummy.txt");

            System.out.println("\n" + "â•".repeat(70));
            System.out.println("EXPERIMENTAL DESIGN - SKIP CONNECTION ANALYSIS WITH STUDENT BASELINES");
            System.out.println("â•".repeat(70));
            System.out.println("Skip Percentages to Test: " + formatSkipArray(SKIP_PERCENTAGES_TO_TEST));
            System.out.println("Parameter Deviation Allowed: Â±" + (MAX_PARAM_DEVIATION * 100) + "%");
            System.out.println("Teachers: 3 architectures (Shallow, Medium, Deep) ~15K params each");
            System.out.println("Students: 9 architectures per class, all matched to class baseline");
            System.out.println("  - First student in each (input, output) = baseline");
            System.out.println("  - Other students adjusted to match baseline parameters");
            System.out.println("Replications: " + NUM_REPLICATIONS + " per configuration");
            System.out.println("Total Experiments: " + SKIP_PERCENTAGES_TO_TEST.length + " skip% Ã— 3 teachers Ã— 9 students Ã— " + 
                             NUM_REPLICATIONS + " reps = " + (SKIP_PERCENTAGES_TO_TEST.length * 3 * 9 * NUM_REPLICATIONS));
            System.out.println("â•".repeat(70) + "\n");

            // PHASE 1: Calculate baseline parameters (first student per class)
            calculateBaselineParameters();
            
            // PHASE 2: Run experiments with each skip percentage
            for (double skipPercent : SKIP_PERCENTAGES_TO_TEST) {
                CURRENT_SKIP_PERCENTAGE = skipPercent;
                
                System.out.println("\n\n" + "â•”".repeat(70));
                System.out.println("â•‘  PHASE 2 - EXPERIMENTAL RUN: SKIP CONNECTIONS = " + String.format("%.1f%%", skipPercent * 100));
                System.out.println("â•š".repeat(70) + "\n");
                
                runExperiment();
            }

            System.out.println("\n" + "â•".repeat(70));
            System.out.println("ALL EXPERIMENTS COMPLETED");
            System.out.println("â•".repeat(70));
            System.out.println("âœ“ Tested skip percentages: " + formatSkipArray(SKIP_PERCENTAGES_TO_TEST));
            System.out.println("âœ“ All students within Â±" + (MAX_PARAM_DEVIATION * 100) + "% of class baseline");
            System.out.println("âœ“ Balanced teacher parameters (~15K each)");
            System.out.println("âœ“ " + NUM_REPLICATIONS + " replications per configuration");
            System.out.println("âœ“ Tests depth + width pattern effects + skip connection effects");
            System.out.println("â•".repeat(70));

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            scan.close();
        }
    }

    private static String formatSkipArray(double[] skips) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < skips.length; i++) {
            sb.append(String.format("%.0f%%", skips[i] * 100));
            if (i < skips.length - 1) sb.append(", ");
        }
        return sb.toString();
    }

    public static void runExperiment() {
        try {
            if (runRealDatasetIfAvailable()) {
                return;
            }
            for (int input : inputsize) {
                for (int output : outputsize) {
                    List<int[]> matchingTeachers = getMatchingTeachers(input, output);

                    if (matchingTeachers.isEmpty()) {
                        System.out.println("No teacher configs for input=" + input + ", output=" + output);
                        continue;
                    }

                    System.out.println("\n" + "=".repeat(60));
                    System.out.println("PROCESSING INPUT=" + input + ", OUTPUT=" + output);
                    System.out.println("Skip Connections: " + String.format("%.1f%%", CURRENT_SKIP_PERCENTAGE * 100) + " (FIXED)");
                    System.out.println("Found " + matchingTeachers.size() + " teacher configuration(s)");
                    System.out.println("=".repeat(60));

                    List<int[]> matchingStudents = getMatchingStudents(input, output);
                    
                    if (matchingStudents.isEmpty()) {
                        System.out.println("No student configs for input=" + input + ", output=" + output);
                        continue;
                    }

                    String classKey = getClassKey(input, output);
                    Integer classBaseline = CLASS_BASELINE_PARAMS.get(classKey);

                    // ITERATE THROUGH EACH TEACHER
                    for (int teacherIdx = 0; teacherIdx < matchingTeachers.size(); teacherIdx++) {
                        int[] T = matchingTeachers.get(teacherIdx);
                        String teacherDepth = getTeacherDepth(T);

                        System.out.println("\n" + "â”€".repeat(60));
                        System.out.println("TEACHER EXPERIMENT " + (teacherIdx + 1) + "/" + matchingTeachers.size());
                        System.out.println("Architecture: " + getConfigLabel(T) + " (" + teacherDepth + ")");
                        System.out.println("â”€".repeat(60));

                        // REPLICATIONS LOOP
                        for (int replication = 0; replication < NUM_REPLICATIONS; replication++) {
                            System.out.println("\n" + "â”Œ".repeat(50));
                            System.out.println("REPLICATION " + (replication + 1) + "/" + NUM_REPLICATIONS);
                            System.out.println("â””".repeat(50));

                            // Set seed for reproducibility
                            long seed = (teacherIdx * 1000 + replication * 100);
                            
                            // Prepare training and validation datasets
                            fnTrain.Tinputs = new double[numdatapoints][input];
                            fnTrain.Ttargets = new double[numdatapoints][output];
                            fnTrain.TinputsValidate = new double[numValdatapoints][input];
                            fnTrain.TtargetsValidate = new double[numValdatapoints][output];

                            fnTrain.Teacher = new network();
                            fnCreate.createNetworkStd(fnTrain.Teacher, T);

                            // Generate balanced training dataset
                            boolean trainingDataBalanced = false;
                            int trainAttempts = 0;
                            int maxAttempts = 100;

                            while (!trainingDataBalanced && trainAttempts++ < maxAttempts) {
                                trainingDataBalanced = fnTeacher.createDataset(fnTrain.Teacher, fnTrain.Tinputs, fnTrain.Ttargets);
                            }

                            if (!trainingDataBalanced) {
                                System.out.println("WARNING: Failed to generate BALANCED training data after " + maxAttempts + " attempts.");
                                continue;
                            }

                            // Generate balanced validation dataset
                            boolean validationDataBalanced = false;
                            int valAttempts = 0;

                            while (!validationDataBalanced && valAttempts++ < maxAttempts) {
                                validationDataBalanced = fnTeacher.createDataset(fnTrain.Teacher, fnTrain.TinputsValidate, fnTrain.TtargetsValidate);
                            }

                            if (!validationDataBalanced) {
                                System.out.println("WARNING: Failed to generate BALANCED validation data after " + maxAttempts + " attempts.");
                                continue;
                            }

                            int teacherParams = countNetworkParameters(fnTrain.Teacher);

                            System.out.println("\n=== TEACHER NETWORK (A) - Rep " + (replication + 1) + " ===");
                            System.out.println("Architecture: " + getConfigLabel(T));
                            System.out.println("Depth: " + teacherDepth + " (" + (T.length - 2) + " hidden layers)");
                            System.out.println("Total Parameters: " + teacherParams);
                            System.out.println("Seed: " + seed);
                            System.out.println("===========================\n");

                            // Test each student configuration
                            for (int configIndex = 0; configIndex < matchingStudents.size(); configIndex++) {
                                int[] baseConfig = matchingStudents.get(configIndex);

                                System.out.println("\n" + "â•".repeat(50));
                                System.out.println("STUDENT " + (configIndex + 1) + "/" + matchingStudents.size() + 
                                                 " | Rep " + (replication + 1) + "/" + NUM_REPLICATIONS);
                                System.out.println("Original Base: " + getConfigLabel(baseConfig));
                                System.out.println("â•".repeat(50));

                                // Use class baseline for ALL students in this class
                                ArchitectureResult result = findMatchingArchitecture(baseConfig, classBaseline);

                                if (result == null || result.config == null) {
                                    System.out.println("âŒ ERROR: Could not find suitable architecture within Â±" + 
                                                     (MAX_PARAM_DEVIATION * 100) + "% of baseline.");
                                    System.out.println("Skipping configuration " + (configIndex + 1) + "\n");
                                    continue;
                                }

                                int[] S = result.config;
                                double SP = result.skipPercentage;
                                int finalParams = result.actualParams;

                                String configLabel = getConfigLabel(S);
                                String studentDepth = getStudentDepth(S);
                                
                                // Determine if this is the baseline config for this class
                                boolean isBaseline = (configIndex == 0);

                                System.out.println("\nâœ… CONFIGURATION READY");
                                System.out.println("Student Depth: " + studentDepth);
                                System.out.println("Final Architecture: " + configLabel);
                                System.out.println("Final Parameters (B): " + finalParams + " (Class Baseline: " + classBaseline + ")");
                                System.out.println("Deviation: " + String.format("%.2f%%", Math.abs(finalParams - classBaseline) / (double) classBaseline * 100));
                                System.out.println("Skip Percentage: " + String.format("%.2f%%", SP * 100) + " (FIXED)");
                                System.out.println("Baseline Student: " + (isBaseline ? "YES âœ“" : "NO (matched to baseline)"));
                                System.out.println("â•".repeat(50));

                                // BASE DIRECTORY with skip percentage at top level
                                String baseDirectoryPath = String.format(
                                    "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\ExperimentSet_SkipAnalysis60.1\\SkipPercent_%02d\\%dinput_%doutput\\Teacher_%s_%s\\Rep%d\\Student_%s_%s\\Config%d\\",
                                    (int)(CURRENT_SKIP_PERCENTAGE * 100),
                                    input, output, 
                                    teacherDepth, getConfigLabel(T).replace(" ", ""),
                                    replication + 1,
                                    studentDepth, configLabel.replace(" ", ""),
                                    configIndex + 1
                                );

                                ensureDirectoryExists(baseDirectoryPath + "dummy.txt");

                                String commonLogFileName = baseDirectoryPath + "common_log.txt";
                                ensureDirectoryExists(commonLogFileName);

                                try (PrintWriter commonWriter = new PrintWriter(new FileWriter(commonLogFileName, true))) {
                                    commonWriter.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
                                    commonWriter.println("SKIP CONNECTION EXPERIMENTAL DESIGN");
                                    commonWriter.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
                                    commonWriter.println("Skip Percentage (FIXED): " + String.format("%.1f%%", CURRENT_SKIP_PERCENTAGE * 100));
                                    commonWriter.println("Teacher Exp: " + (teacherIdx + 1) + " | Student: " + (configIndex + 1) + " | Rep: " + (replication + 1));
                                    commonWriter.println("Random Seed: " + seed);
                                    commonWriter.println();
                                    
                                    // Parameter matching validation section
                                    commonWriter.println("â•â•â• PARAMETER MATCHING VALIDATION (CLASS BASELINE) â•â•â•");
                                    commonWriter.println("Class Baseline Parameters (0% skip):  " + classBaseline);
                                    commonWriter.println("Current Parameters:                   " + finalParams);
                                    commonWriter.println("Difference:                           " + (finalParams - classBaseline));
                                    commonWriter.println("Deviation:                            " + 
                                                       String.format("%.2f%%", Math.abs(finalParams - classBaseline) / (double) classBaseline * 100));
                                    commonWriter.println("Is Baseline Student:                  " + (isBaseline ? "YES âœ“" : "NO"));
                                    
                                    // Visual indicator
                                    double deviation = Math.abs(finalParams - classBaseline) / (double) classBaseline;
                                    if (deviation <= 0.05) {
                                        commonWriter.println("Status:                               âœ“âœ“ EXCELLENT (within 5%)");
                                    } else if (deviation <= MAX_PARAM_DEVIATION) {
                                        commonWriter.println("Status:                               âœ“ ACCEPTABLE (within " + (MAX_PARAM_DEVIATION * 100) + "%)");
                                    } else {
                                        commonWriter.println("Status:                               âŒ OUT OF RANGE (>" + (MAX_PARAM_DEVIATION * 100) + "%)");
                                    }
                                    commonWriter.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
                                    commonWriter.println();
                                    
                                    commonWriter.println("  Input Size: " + input);
                                    commonWriter.println("  Output Size: " + output);
                                    commonWriter.println("  Class Key: " + classKey);
                                    commonWriter.println();
                                    commonWriter.println("  Teacher Architecture: " + getConfigLabel(T) + " (" + teacherDepth + ")");
                                    commonWriter.println("  Teacher Parameters (A): " + teacherParams);
                                    commonWriter.println();
                                    commonWriter.println("  Student Original: " + getConfigLabel(baseConfig));
                                    commonWriter.println("  Student Final: " + configLabel + " (" + studentDepth + ")");
                                    commonWriter.println("  Student Parameters (B): " + finalParams);
                                    commonWriter.println("  Class Baseline (0% skip): " + classBaseline);
                                    commonWriter.println("  Skip Percentage (FIXED): " + String.format("%.2f%%", SP * 100));
                                    commonWriter.println();

                                    fnTrain.Student = new network();
                                    fnCreate.createNetwork(fnTrain.Student, S, SP, commonWriter);

                                    int studentParams = countNetworkParameters(fnTrain.Student);

                                    commonWriter.println("\n=== PARAMETER VERIFICATION ===");
                                    commonWriter.println("Teacher Parameters (A): " + teacherParams);
                                    commonWriter.println("Class Baseline Parameters (0% skip): " + classBaseline);
                                    commonWriter.println("Actual Student Parameters (B): " + studentParams);
                                    commonWriter.println("Match vs Class Baseline: " + (Math.abs(studentParams - classBaseline) <= classBaseline * MAX_PARAM_DEVIATION ? "âœ“ YES" : "â—‹ CLOSE"));
                                    commonWriter.println("Difference from Class Baseline: " + Math.abs(studentParams - classBaseline));
                                    commonWriter.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n");

                                    int teacherLayers = countLayers(fnTrain.Teacher);
                                    int studentLayers = countLayers(fnTrain.Student);

                                    commonWriter.println("Teacher Layers: " + teacherLayers + " | Student Layers: " + studentLayers);
                                    commonWriter.println();

                                    fnCreate.verifyOutputLayerConnections(fnTrain.Student, commonWriter);
                                    fnCreate.verifyOutputLayerConnections(fnTrain.Teacher, commonWriter);
                                } catch (IOException e) {
                                    System.out.println("Error writing to common log file: " + e.getMessage());
                                }

                                trainStandard(baseDirectoryPath, input, output, S, SP, replication + 1, classBaseline);
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static boolean runRealDatasetIfAvailable() {
        try {
            double[][] inputs = data.getInputData();
            double[][] targets = data.getTargetData();
            if (inputs == null || targets == null || inputs.length == 0 || targets.length == 0) {
                return false;
            }
            int input = inputs[0].length;
            int output = targets[0].length;
            int n = inputs.length;
            int trainCount = Math.max(1, (int) Math.floor(n * 0.8));
            int valCount = n - trainCount;
            fnTrain.Tinputs = new double[trainCount][input];
            fnTrain.Ttargets = new double[trainCount][output];
            fnTrain.TinputsValidate = new double[valCount][input];
            fnTrain.TtargetsValidate = new double[valCount][output];
            for (int i = 0; i < trainCount; i++) {
                System.arraycopy(inputs[i], 0, fnTrain.Tinputs[i], 0, input);
                System.arraycopy(targets[i], 0, fnTrain.Ttargets[i], 0, output);
            }
            for (int i = 0; i < valCount; i++) {
                System.arraycopy(inputs[trainCount + i], 0, fnTrain.TinputsValidate[i], 0, input);
                System.arraycopy(targets[trainCount + i], 0, fnTrain.TtargetsValidate[i], 0, output);
            }
            int[] S = {input, 8, output};
            String logPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\Real Data\\Set2\\training_and_evaluation_log.txt";
            ensureDirectoryExists(logPath);
            try (PrintWriter writer = new PrintWriter(new FileWriter(logPath, true))) {
                writer.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
                writer.println("Real Dataset Training");
                writer.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
                writer.println("  Input Size: " + input);
                writer.println("  Output Size: " + output);
                writer.println("  Architecture: " + getConfigLabel(S));
                writer.println("  Skip Percentage (FIXED): " + String.format("%.2f%%", CURRENT_SKIP_PERCENTAGE * 100));
                writer.println();
                fnTrain.Student = new network();
                fnCreate.createNetwork(fnTrain.Student, S, 0.0, writer);
                fnTrain.Train(numepochs, input, output, 0, 0.0, writer);
                writer.println();
                writer.println("Training completed on real dataset");
                writer.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    // Simplified architecture matching that uses actual network creation
    private static ArchitectureResult findMatchingArchitecture(int[] baseConfig, int baseline) {
        final int MAX_ITERATIONS = 100;
        int minParams = (int) Math.floor(baseline * (1 - MAX_PARAM_DEVIATION));
        int maxParams = (int) Math.ceil(baseline * (1 + MAX_PARAM_DEVIATION));

        System.out.println("\nâ•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—");
        System.out.println("â•‘  PARAMETER MATCHING VALIDATION          â•‘");
        System.out.println("â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
        System.out.println("Class Baseline:      " + baseline + " params");
        System.out.println("Acceptable range:    " + minParams + " - " + maxParams + " params");
        System.out.println("Current skip %:      " + (CURRENT_SKIP_PERCENTAGE * 100) + "%");
        System.out.println();

        // First, test the base configuration as-is
        int baseParams = createAndCountParams(baseConfig, CURRENT_SKIP_PERCENTAGE);
        if (baseParams >= minParams && baseParams <= maxParams) {
            System.out.println("âœ… BASE CONFIGURATION MATCHES!");
            System.out.println("Architecture: " + architectureToString(baseConfig));
            System.out.println("Parameters: " + baseParams);
            return new ArchitectureResult(baseConfig, CURRENT_SKIP_PERCENTAGE, baseParams);
        }

        // If base doesn't match, try adjustments by scaling ALL hidden layers proportionally
        int[] current = baseConfig.clone();
        int lastParams = baseParams;
        double lastRatio = 1.0;
        
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            int currentParams = createAndCountParams(current, CURRENT_SKIP_PERCENTAGE);
            
            if (currentParams >= minParams && currentParams <= maxParams) {
                double deviation = Math.abs(currentParams - baseline) / (double) baseline * 100;
                System.out.println("âœ… MATCH FOUND at iteration " + (iteration + 1) + "!");
                System.out.println("Architecture: " + architectureToString(current));
                System.out.println("Parameters: " + currentParams);
                System.out.println("Deviation: " + String.format("%.2f%%", deviation));
                return new ArchitectureResult(current, CURRENT_SKIP_PERCENTAGE, currentParams);
            }
            
            System.out.println("Iteration " + (iteration + 1) + ": " + 
                             architectureToString(current) + " â†’ " + currentParams + 
                             " params (target: " + minParams + "-" + maxParams + ")");
            
            // Calculate scaling ratio needed to reach baseline
            double targetRatio = (double) baseline / currentParams;
            
            // Scale ALL hidden layers proportionally
            if (current.length > 2) {
                for (int i = 1; i < current.length - 1; i++) {
                    int newSize = Math.max(2, (int) Math.round(current[i] * targetRatio));
                    current[i] = newSize;
                }
            }
            
            // Safety check: if we're not making progress, try different approach
            if (Math.abs(currentParams - lastParams) < 5 && iteration > 10) {
                System.out.println("  (Converged to stable point, unable to reach target range)");
                break;
            }
            
            lastParams = currentParams;
            lastRatio = targetRatio;
        }
        
        System.out.println("âŒ Failed to find matching architecture after " + MAX_ITERATIONS + " iterations");
        return null;
    }

    // Helper method to create network and count actual parameters
    private static int createAndCountParams(int[] config, double skipPercent) {
        try {
            network testNet = new network();
            String tempLog = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\temp_param_check.txt";
            ensureDirectoryExists(tempLog);
            
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempLog))) {
                fnCreate.createNetwork(testNet, config, skipPercent, writer);
                return countNetworkParameters(testNet);
            }
        } catch (Exception e) {
            System.out.println("Error creating test network: " + e.getMessage());
            return -1;
        }
    }

    private static List<int[]> getMatchingTeachers(int input, int output) {
        List<int[]> matching = new ArrayList<>();
        for (int[] config : teacherConfigs) {
            if (config[0] == input && config[config.length - 1] == output) {
                matching.add(config);
            }
        }
        return matching;
    }

    private static List<int[]> getMatchingStudents(int input, int output) {
        List<int[]> matching = new ArrayList<>();
        for (int[] config : studentConfigs) {
            if (config[0] == input && config[config.length - 1] == output) {
                matching.add(config);
            }
        }
        return matching;
    }

    private static String getTeacherDepth(int[] config) {
        int hiddenLayers = config.length - 2;
        if (hiddenLayers == 1) return "Shallow";
        if (hiddenLayers == 2) return "Medium";
        if (hiddenLayers == 3) return "Deep";
        return "Custom";
    }

        private static String getStudentDepth(int[] config) {
            int hiddenLayers = config.length - 2;
            if (hiddenLayers <= 3) return "Shallow (â‰¤3L)";
            if (hiddenLayers <= 8) return "Medium (4-8L)";
            if (hiddenLayers <= 20) return "Deep (9-20L)";
            return "Very Deep (20+L)";
        }

    private static void trainStandard(String baseDirectoryPath, int input, int output, int[] S, double SP, int replication, int baseline) {
        String logFileName = baseDirectoryPath + "training_log.txt";
        ensureDirectoryExists(logFileName);

        try (PrintWriter writer = new PrintWriter(new FileWriter(logFileName, true))) {
            writer.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
            writer.println("Standard Training (SGD) - Replication " + replication);
            writer.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
            writer.println("  Input Size: " + input);
            writer.println("  Output Size: " + output);
            writer.println("  Architecture: " + getConfigLabel(S));
            writer.println("  Skip Percentage (FIXED): " + String.format("%.2f%%", SP * 100));
            writer.println("  Class Baseline Parameters (0% skip): " + baseline);
            writer.println();

            fnTrain.Student = new network();
            fnCreate.createNetwork(fnTrain.Student, S, SP, writer);

            int studentParams = countNetworkParameters(fnTrain.Student);
            double deviation = Math.abs(studentParams - baseline) / (double) baseline * 100;
            
            writer.println("Student Parameters (B): " + studentParams);
            writer.println("Deviation from Class Baseline: " + String.format("%.2f%%", deviation));
            writer.println();

            fnTrain.Train(numepochs, input, output, 0, SP, writer);

            writer.println();
            writer.println("Training completed for replication " + replication);
            writer.println("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
        } catch (IOException e) {
            System.out.println("Error writing to training log file: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static int countLayers(network net) {
        int count = 0;
        layernode current = net.inputlayernode;
        while (current != null) {
            count++;
            current = current.next;
        }
        return count;
    }

    public static int countNetworkParameters(network net) {
        int totalParams = 0;
        layernode layer = net.inputlayernode;

        while (layer != null) {
            node n = layer.firstnode;
            while (n != null) {
                if (layer != net.inputlayernode) {
                    totalParams++;
                }
                edge e = n.firstedge;
                while (e != null) {
                    totalParams++;
                    e = e.next;
                }
                n = n.next;
            }
            layer = layer.next;
        }
        return totalParams;
    }

    private static String getConfigLabel(int[] config) {
        StringBuilder label = new StringBuilder();
        for (int i = 0; i < config.length; i++) {
            label.append(config[i]);
            if (i == 0) {
                label.append("n");
            } else if (i == config.length - 1) {
                label.append("o");
            } else {
                label.append("n");
            }
            if (i < config.length - 1) {
                label.append("-");
            }
        }
        return label.toString();
    }

    static class ArchitectureResult {
        int[] config;
        double skipPercentage;
        int actualParams;

        ArchitectureResult(int[] config, double skipPercentage, int actualParams) {
            this.config = config;
            this.skipPercentage = skipPercentage;
            this.actualParams = actualParams;
        }
    }

    public static String architectureToString(int[] config) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < config.length; i++) {
            sb.append(config[i]);
            if (i == 0) sb.append("n");
            else if (i == config.length - 1) sb.append("o");
            else sb.append("n");
            if (i < config.length - 1) sb.append("-");
        }
        return sb.toString();
    }
}