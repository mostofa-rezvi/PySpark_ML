import os
from fpdf import FPDF

class AssignmentReport(FPDF):
    def header(self):
        if self.page_no() == 1:
            return  # Skip header on the cover page
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, 'PySpark Machine Learning Assignment Report', 0, 0, 'L')
        # Add a subtle page number in the header right side
        self.cell(0, 10, f'Page {self.page_no()}', 0, 1, 'R')
        self.ln(2)
        # Draw a horizontal line under the header
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.5)
        self.line(15, 23, 195, 23)
        self.ln(5)
        
    def footer(self):
        if self.page_no() == 1:
            return  # Skip footer on the cover page
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        # Draw a line above footer
        self.set_draw_color(220, 220, 220)
        self.set_line_width(0.3)
        self.line(15, 282, 195, 282)
        self.cell(0, 10, 'PySpark ML Assignment | Supervised & Unsupervised Learning', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'R')

    def add_section_header(self, text):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 58, 138)  # Primary color: Navy Blue
        self.cell(0, 10, text, 0, 1, 'L')
        self.ln(2)

    def add_subsection_header(self, text):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50, 80, 150)
        self.cell(0, 8, text, 0, 1, 'L')
        self.ln(1)

    def add_body_text(self, text, align='J'):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(180, 6, text, 0, align)
        self.ln(4)

def main():
    print("--- Starting Report Generation ---")
    
    pdf = AssignmentReport()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ------------------ PAGE 1: COVER PAGE ------------------
    pdf.add_page()
    
    # Add a colored accent sidebar/top block
    pdf.set_fill_color(30, 58, 138) # Deep Navy
    pdf.rect(0, 0, 210, 85, 'F')
    
    # Title Text inside top band
    pdf.set_y(25)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, 'Machine Learning with PySpark', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 14)
    pdf.cell(0, 10, 'Supervised & Unsupervised Case Studies', 0, 1, 'C')
    
    # White space
    pdf.set_y(100)
    
    # Subtitle or Course info
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, 'ASSIGNMENT REPORT', 0, 1, 'C')
    
    # Decorative line
    pdf.set_draw_color(30, 58, 138)
    pdf.set_line_width(1.5)
    pdf.line(50, 115, 160, 115)
    
    # Student and Project Details
    pdf.set_y(140)
    
    details = [
        ("Course Task:", "PySpark MLlib Assignments (Part A & B)"),
        ("Student Name:", "Mostofa Aminur Rashid"),
        ("Student ID:", "2506107"),
        ("Platform:", "Apache Spark 4.1.2 (via PySpark)"),
        ("Java Version:", "JDK 17 LTS (for local JVM execution)"),
        ("Submission Date:", "July 6, 2026")
    ]
    
    for label, val in details:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(50, 8, label, 0, 0, 'R')
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(70, 70, 70)
        pdf.cell(10, 8, '', 0, 0) # spacer
        pdf.cell(120, 8, val, 0, 1, 'L')
    
    # Footer-like note on cover page
    pdf.set_y(240)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, 'Generated programmatically using Python and PySpark MLlib.', 0, 1, 'C')
    
    # ------------------ PAGE 2: PART A - INTRO & METRICS ------------------
    pdf.add_page()
    pdf.add_section_header('Part A: Supervised Learning - Titanic Survival Prediction')
    
    intro_p1 = (
        "The objective of this task is to develop a supervised machine learning model using PySpark "
        "to predict passenger survival aboard the RMS Titanic based on demographic and ticketing data. "
        "The dataset includes information like ticket class, gender, age, family members aboard (siblings/parents), "
        "ticket fare, cabin, and port of embarkation."
    )
    pdf.add_body_text(intro_p1)
    
    pdf.add_subsection_header('Data Preprocessing & Engineering Decisions')
    prep_p1 = (
        "1. Missing Values: Imputed the missing values in 'Age' with the median age (28.0) using PySpark's approxQuantile "
        "method. Imputed the missing 'Embarked' values with the mode ('S'). Dropped the 'Cabin' column entirely since over "
        "70% of its values were null.\n"
        "2. Feature Selection: Kept 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', and 'Embarked'. String identifier "
        "columns ('PassengerId', 'Name', and 'Ticket') were dropped as they carry no predictive signal.\n"
        "3. Categorical Encoding: Built a Pipeline where 'Sex' and 'Embarked' were indexed using StringIndexer and then "
        "converted into One-Hot Encoded (OHE) vectors via OneHotEncoder.\n"
        "4. Feature Assembly: Combined all numerical and OHE vectors into a single feature vector using VectorAssembler "
        "to feed into the classifier."
    )
    pdf.add_body_text(prep_p1)
    
    pdf.add_subsection_header('Model Selection & Evaluation')
    eval_p1 = (
        "We chose a Random Forest Classifier (100 trees, maximum depth of 5) for this binary classification task. "
        "The dataset was split into 80% training and 20% testing sets. We evaluated the model using "
        "standard classification metrics on the test dataset:"
    )
    pdf.add_body_text(eval_p1)
    
    # Draw Metrics Table
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 58, 138)
    
    pdf.cell(90, 8, 'Evaluation Metric', 1, 0, 'C', True)
    pdf.cell(90, 8, 'Obtained Test Value', 1, 1, 'C', True)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    
    metrics = [
        ("Accuracy", "82.76%"),
        ("Weighted Precision", "83.39%"),
        ("Weighted Recall", "82.76%"),
        ("Weighted F1-score", "82.35%"),
        ("Area Under ROC (AUC)", "89.12%")
    ]
    
    bg = False
    for metric, val in metrics:
        pdf.set_fill_color(240, 243, 250) if bg else pdf.set_fill_color(255, 255, 255)
        pdf.cell(90, 8, metric, 1, 0, 'L', True)
        pdf.cell(90, 8, val, 1, 1, 'C', True)
        bg = not bg
        
    pdf.ln(5)
    
    # ------------------ PAGE 3: PART A - PLOTS & DISCUSSION ------------------
    pdf.add_page()
    pdf.add_section_header('Part A: Visualizations & Discussion')
    
    # Add Confusion Matrix and Feature Importances side-by-side or stacked
    # Since they are 150 dpi, let's stack them vertically with captions
    cm_img_path = os.path.join("outputs", "titanic_confusion_matrix.png")
    fi_img_path = os.path.join("outputs", "titanic_feature_importances.png")
    
    if os.path.exists(cm_img_path):
        pdf.image(cm_img_path, x=20, y=35, w=80)
        
    if os.path.exists(fi_img_path):
        pdf.image(fi_img_path, x=105, y=32, w=90)
        
    # Spacer
    pdf.set_y(105)
    pdf.ln(5)
    
    pdf.add_subsection_header('Discussion on Supervised Results')
    disc_p1 = (
        "The Random Forest model demonstrates excellent performance on the Titanic test set, achieving an Accuracy of "
        "82.76% and a high AUC of 89.12%. Analyzing the confusion matrix, the model correctly predicted 78 survivors/non-survivors "
        "in class 0 (True Negatives) and 42 in class 1 (True Positives), with only 6 False Positives. This suggests a high precision "
        "(83.39%), making the classifier highly reliable when predicting survival.\n\n"
        "Looking at the Feature Importances plot, the passenger's Gender (Sex_Male) and ticket Class (Pclass) emerge as the "
        "strongest predictors of survival. This is highly aligned with historical accounts of the disaster, where women and children "
        "were prioritized for lifeboats, and upper-class cabins were located closer to the boat deck. Demographic factors like Age "
        "and Fare also contributed moderately to the model's decisions."
    )
    pdf.add_body_text(disc_p1)
    
    # ------------------ PAGE 4: PART B - MALL CLUSTERING ------------------
    pdf.add_page()
    pdf.add_section_header('Part B: Unsupervised Learning - Mall Customer Segmentation')
    
    partb_intro = (
        "In this task, we perform market segmentation on the Mall Customer dataset using unsupervised learning. "
        "Unlike supervised learning, there are no predefined labels. The goal is to discover natural "
        "customer groupings based on Age, Annual Income (k$), and Spending Score (1-100) using the K-Means algorithm."
    )
    pdf.add_body_text(partb_intro)
    
    pdf.add_subsection_header('Clustering Pipeline & Scaling')
    partb_prep = (
        "1. Feature Selection: We selected the three numerical attributes: Age, Annual Income, and Spending Score.\n"
        "2. Vector Assembly: Assembled features into a dense vector using VectorAssembler.\n"
        "3. StandardScaler: Because K-Means is a distance-based algorithm (relying on Euclidean distance), features "
        "with larger scales (like Annual Income) would dominate features with smaller scales (like Age). We applied "
        "StandardScaler to standardize the data to have a mean of 0 and unit variance."
    )
    pdf.add_body_text(partb_prep)
    
    pdf.add_subsection_header('K-Means Cluster Centers Interpretation')
    partb_centers = (
        "We ran the K-Means algorithm for two different configurations: k = 3 and k = 5. Below are the resulting "
        "cluster centroids mapped back to their original dimensions:"
    )
    pdf.add_body_text(partb_centers)
    
    # Draw Cluster Centers Table
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 58, 138)
    
    pdf.cell(10, 8, 'k', 1, 0, 'C', True)
    pdf.cell(65, 8, 'Cluster', 1, 0, 'C', True)
    pdf.cell(20, 8, 'Avg. Age', 1, 0, 'C', True)
    pdf.cell(40, 8, 'Avg. Annual Income (k$)', 1, 0, 'C', True)
    pdf.cell(45, 8, 'Avg. Spending Score (1-100)', 1, 1, 'C', True)
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(50, 50, 50)
    
    # Data from outputs/mall_results.txt
    # k=3:
    # Cluster 0: Age=33.42, Income=89.02, Spending Score=75.58
    # Cluster 1: Age=25.14, Income=43.80, Spending Score=55.81
    # Cluster 2: Age=51.18, Income=58.27, Spending Score=33.70
    # k=5:
    # Cluster 0: Age=32.88, Income=86.10, Spending Score=81.53
    # Cluster 1: Age=55.54, Income=47.95, Spending Score=41.89
    # Cluster 2: Age=44.39, Income=89.77, Spending Score=18.48
    # Cluster 3: Age=27.06, Income=51.98, Spending Score=41.04
    # Cluster 4: Age=25.52, Income=26.30, Spending Score=78.57
    
    rows = [
        ("3", "Cluster 0", "33.42", "89.02", "75.58"),
        ("3", "Cluster 1", "25.14", "43.80", "55.81"),
        ("3", "Cluster 2", "51.18", "58.27", "33.70"),
        ("5", "Cluster 0 (Elite)", "32.88", "86.10", "81.53"),
        ("5", "Cluster 1 (Older, Low Spend)", "55.54", "47.95", "41.89"),
        ("5", "Cluster 2 (High Income, Low Spend)", "44.39", "89.77", "18.48"),
        ("5", "Cluster 3 (Young, Mod Spend)", "27.06", "51.98", "41.04"),
        ("5", "Cluster 4 (Low Income, High Spend)", "25.52", "26.30", "78.57"),
    ]
    
    bg = False
    for r in rows:
        pdf.set_fill_color(240, 243, 250) if bg else pdf.set_fill_color(255, 255, 255)
        pdf.cell(10, 7, r[0], 1, 0, 'C', True)
        pdf.cell(65, 7, r[1], 1, 0, 'L', True)
        pdf.cell(20, 7, r[2], 1, 0, 'C', True)
        pdf.cell(40, 7, r[3], 1, 0, 'C', True)
        pdf.cell(45, 7, r[4], 1, 1, 'C', True)
        bg = not bg
        
    pdf.ln(5)
    
    # ------------------ PAGE 5: PART B - PLOTS & DISCUSSION ------------------
    pdf.add_page()
    pdf.add_section_header('Part B: Visualizations & Discussion')
    
    k3_img_path = os.path.join("outputs", "mall_clusters_k3.png")
    k5_img_path = os.path.join("outputs", "mall_clusters_k5.png")
    
    if os.path.exists(k3_img_path):
        pdf.image(k3_img_path, x=15, y=35, w=85)
        
    if os.path.exists(k5_img_path):
        pdf.image(k5_img_path, x=105, y=35, w=85)
        
    pdf.set_y(105)
    pdf.ln(5)
    
    pdf.add_subsection_header('Discussion on Unsupervised Clustering')
    disc_p2 = (
        "The K-Means clustering algorithm partitioned the customer base into distinct archetypes. Comparing the results "
        "between k = 3 and k = 5, we observe that k = 5 provides much more actionable insights for business planning:\n\n"
        "1. For k = 3: The model splits customers into high-income/high-spend (Cluster 0), younger moderate-spend (Cluster 1), "
        "and older lower-spend (Cluster 2). While useful, this grouping is too broad.\n\n"
        "2. For k = 5: We discover key market segments:\n"
        "   - Elite Target Group (Cluster 0): High income and high spending score. These are highly profitable customers "
        "to target with premium promotions.\n"
        "   - Careful / High Income, Low Spend (Cluster 2): Customers with substantial resources who are conservative in "
        "their spending. The mall can design targeted campaigns to convert their interest.\n"
        "   - Sensible / Low Income, High Spend (Cluster 4): Younger customers who spend heavily despite lower income. "
        "Marketing can offer them credit options or student-friendly deals.\n"
        "   - Average / Young, Moderate Spend (Cluster 3) & Standard / Older, Low Spend (Cluster 1): Solid baseline segments."
    )
    pdf.add_body_text(disc_p2)
    
    # Output file
    output_pdf_path = "PySpark_ML_Assignment_Report.pdf"
    pdf.output(output_pdf_path)
    print(f"Report generated successfully at: {output_pdf_path}")

if __name__ == "__main__":
    main()
