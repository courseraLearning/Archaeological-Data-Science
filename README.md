# Archaeological-Data-Science
# Payments Application Data Archaeology - Complete Implementation Guide

## Phase 1: Initial Data Discovery & Profiling

### 1.1 Data Inventory & Structure Analysis

```python
# Hive/Spark SQL for initial data discovery
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder \
    .appName("PaymentsDataArchaeology") \
    .enableHiveSupport() \
    .getOrCreate()

# 1. Catalog all tables and their metadata
def discover_tables():
    """Discover all tables in the payments database"""
    tables_query = """
    SHOW TABLES IN payments_db
    """
    tables_df = spark.sql(tables_query)
    return tables_df.collect()

# 2. Analyze table structure and metadata
def analyze_table_structure(table_name):
    """Analyze structure of individual tables"""
    structure_query = f"""
    DESCRIBE EXTENDED {table_name}
    """
    structure_df = spark.sql(structure_query)
    
    # Get table statistics
    stats_query = f"""
    ANALYZE TABLE {table_name} COMPUTE STATISTICS FOR COLUMNS
    """
    spark.sql(stats_query)
    
    # Basic table info
    count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
    row_count = spark.sql(count_query).collect()[0][0]
    
    return {
        'table_name': table_name,
        'row_count': row_count,
        'schema': structure_df.collect(),
        'columns': [row.col_name for row in structure_df.collect() if row.col_name not in ['', '# Detailed Table Information', '# Storage Information']]
    }

# 3. Identify potential relationships
def identify_relationships(tables_info):
    """Identify potential foreign key relationships"""
    relationships = []
    
    for table1 in tables_info:
        for table2 in tables_info:
            if table1['table_name'] != table2['table_name']:
                # Look for common column patterns
                common_cols = set(table1['columns']) & set(table2['columns'])
                if common_cols:
                    relationships.append({
                        'table1': table1['table_name'],
                        'table2': table2['table_name'],
                        'common_columns': list(common_cols)
                    })
    
    return relationships
```

### 1.2 Automated Data Profiling

```python
def comprehensive_data_profile(table_name):
    """Generate comprehensive data profile for a table"""
    df = spark.sql(f"SELECT * FROM {table_name}")
    
    profile_results = {}
    
    # Basic statistics
    profile_results['basic_stats'] = {
        'row_count': df.count(),
        'column_count': len(df.columns),
        'duplicate_rows': df.count() - df.dropDuplicates().count()
    }
    
    # Column-level profiling
    column_profiles = {}
    
    for col in df.columns:
        col_type = dict(df.dtypes)[col]
        
        column_profile = {
            'data_type': col_type,
            'null_count': df.filter(col(col).isNull()).count(),
            'null_percentage': (df.filter(col(col).isNull()).count() / df.count()) * 100,
            'distinct_count': df.select(col).distinct().count(),
            'cardinality_ratio': df.select(col).distinct().count() / df.count()
        }
        
        # Type-specific profiling
        if col_type in ['int', 'bigint', 'double', 'float']:
            stats = df.select(
                min(col).alias('min'),
                max(col).alias('max'),
                mean(col).alias('mean'),
                stddev(col).alias('stddev')
            ).collect()[0]
            
            column_profile.update({
                'min_value': stats['min'],
                'max_value': stats['max'],
                'mean_value': stats['mean'],
                'std_dev': stats['stddev']
            })
            
            # Detect outliers using IQR method
            quantiles = df.select(
                expr(f"percentile_approx({col}, 0.25)").alias('q1'),
                expr(f"percentile_approx({col}, 0.75)").alias('q3')
            ).collect()[0]
            
            if quantiles['q1'] and quantiles['q3']:
                iqr = quantiles['q3'] - quantiles['q1']
                lower_bound = quantiles['q1'] - 1.5 * iqr
                upper_bound = quantiles['q3'] + 1.5 * iqr
                
                outlier_count = df.filter(
                    (col(col) < lower_bound) | (col(col) > upper_bound)
                ).count()
                
                column_profile['outlier_count'] = outlier_count
                column_profile['outlier_percentage'] = (outlier_count / df.count()) * 100
        
        elif col_type == 'string':
            # String-specific analysis
            length_stats = df.select(
                min(length(col)).alias('min_length'),
                max(length(col)).alias('max_length'),
                mean(length(col)).alias('avg_length')
            ).collect()[0]
            
            column_profile.update({
                'min_length': length_stats['min_length'],
                'max_length': length_stats['max_length'],
                'avg_length': length_stats['avg_length']
            })
            
            # Pattern analysis for codes
            column_profile['patterns'] = analyze_string_patterns(df, col)
        
        column_profiles[col] = column_profile
    
    profile_results['column_profiles'] = column_profiles
    return profile_results

def analyze_string_patterns(df, column_name):
    """Analyze string patterns to identify codes and formats"""
    patterns = {}
    
    # Sample values for pattern analysis
    sample_values = df.select(column_name).filter(
        col(column_name).isNotNull()
    ).distinct().limit(100).collect()
    
    values = [row[column_name] for row in sample_values if row[column_name]]
    
    if not values:
        return patterns
    
    # Common patterns
    pattern_checks = {
        'numeric_only': r'^\d+$',
        'alphanumeric': r'^[A-Za-z0-9]+$',
        'contains_dash': r'.*-.*',
        'contains_underscore': r'.*_.*',
        'all_caps': r'^[A-Z]+$',
        'date_like': r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$',
        'currency_code': r'^[A-Z]{3}$',
        'country_code': r'^[A-Z]{2}$',
        'swift_code': r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$'
    }
    
    import re
    for pattern_name, pattern_regex in pattern_checks.items():
        matching_count = sum(1 for val in values if re.match(pattern_regex, str(val)))
        if matching_count > 0:
            patterns[pattern_name] = {
                'count': matching_count,
                'percentage': (matching_count / len(values)) * 100
            }
    
    return patterns
```

## Phase 2: Pattern Recognition & Business Logic Discovery

### 2.1 Temporal Analysis

```python
def temporal_analysis(table_name, date_columns):
    """Analyze temporal patterns in the data"""
    df = spark.sql(f"SELECT * FROM {table_name}")
    
    temporal_insights = {}
    
    for date_col in date_columns:
        # Convert to proper date format if needed
        df_with_date = df.withColumn(
            f"{date_col}_parsed",
            to_date(col(date_col))
        )
        
        # Daily patterns
        daily_pattern = df_with_date.groupBy(
            dayofweek(f"{date_col}_parsed").alias('day_of_week')
        ).count().orderBy('day_of_week')
        
        # Monthly patterns
        monthly_pattern = df_with_date.groupBy(
            month(f"{date_col}_parsed").alias('month')
        ).count().orderBy('month')
        
        # Hourly patterns (if timestamp available)
        if 'timestamp' in str(df.schema[date_col].dataType).lower():
            hourly_pattern = df_with_date.groupBy(
                hour(col(date_col)).alias('hour')
            ).count().orderBy('hour')
            temporal_insights[f'{date_col}_hourly'] = hourly_pattern.collect()
        
        # Data freshness analysis
        latest_date = df_with_date.select(
            max(f"{date_col}_parsed").alias('latest')
        ).collect()[0]['latest']
        
        earliest_date = df_with_date.select(
            min(f"{date_col}_parsed").alias('earliest')
        ).collect()[0]['earliest']
        
        temporal_insights[date_col] = {
            'daily_pattern': daily_pattern.collect(),
            'monthly_pattern': monthly_pattern.collect(),
            'date_range': {
                'earliest': earliest_date,
                'latest': latest_date,
                'span_days': (latest_date - earliest_date).days if latest_date and earliest_date else None
            }
        }
    
    return temporal_insights

def detect_batch_processing_windows(table_name, timestamp_col):
    """Detect batch processing patterns"""
    df = spark.sql(f"SELECT * FROM {table_name}")
    
    # Analyze record creation patterns by hour
    hourly_volume = df.groupBy(
        hour(col(timestamp_col)).alias('hour')
    ).count().orderBy('hour')
    
    # Identify potential batch windows (high volume periods)
    hourly_data = hourly_volume.collect()
    volumes = [row['count'] for row in hourly_data]
    mean_volume = np.mean(volumes)
    std_volume = np.std(volumes)
    
    batch_windows = []
    for row in hourly_data:
        if row['count'] > mean_volume + 2 * std_volume:
            batch_windows.append({
                'hour': row['hour'],
                'volume': row['count'],
                'volume_ratio': row['count'] / mean_volume
            })
    
    return {
        'hourly_pattern': hourly_data,
        'batch_windows': batch_windows,
        'statistics': {
            'mean_volume': mean_volume,
            'std_volume': std_volume
        }
    }
```

### 2.2 Entity Relationship Discovery

```python
def discover_entity_relationships(tables_info):
    """Use statistical methods to discover entity relationships"""
    relationships = {}
    
    for table_info in tables_info:
        table_name = table_info['table_name']
        df = spark.sql(f"SELECT * FROM {table_name}")
        
        # Convert to Pandas for correlation analysis (sample for large datasets)
        if df.count() > 100000:
            df_sample = df.sample(0.1)  # Sample 10% for large datasets
        else:
            df_sample = df
        
        pandas_df = df_sample.toPandas()
        
        # Numeric correlations
        numeric_cols = pandas_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = pandas_df[numeric_cols].corr()
            
            # Find strong correlations (> 0.8 or < -0.8)
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        strong_correlations.append({
                            'col1': correlation_matrix.columns[i],
                            'col2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            relationships[table_name] = {
                'correlations': strong_correlations,
                'correlation_matrix': correlation_matrix.to_dict()
            }
    
    return relationships

def association_rule_mining(table_name, categorical_columns):
    """Find association rules in categorical data"""
    from itertools import combinations
    
    df = spark.sql(f"SELECT * FROM {table_name}")
    
    # Sample for performance
    df_sample = df.sample(0.1) if df.count() > 50000 else df
    
    association_rules = []
    
    # Generate combinations of categorical columns
    for col_pair in combinations(categorical_columns, 2):
        col1, col2 = col_pair
        
        # Create contingency table
        contingency = df_sample.groupBy(col1, col2).count()
        contingency_pd = contingency.toPandas()
        
        if len(contingency_pd) > 1:
            # Calculate support, confidence, and lift
            total_records = df_sample.count()
            
            for _, row in contingency_pd.iterrows():
                support = row['count'] / total_records
                
                # Calculate confidence: P(col2|col1)
                col1_count = df_sample.filter(col(col1) == row[col1]).count()
                confidence = row['count'] / col1_count if col1_count > 0 else 0
                
                # Calculate lift
                col2_count = df_sample.filter(col(col2) == row[col2]).count()
                expected = (col1_count * col2_count) / total_records
                lift = row['count'] / expected if expected > 0 else 0
                
                if support > 0.01 and confidence > 0.5 and lift > 1.2:  # Thresholds
                    association_rules.append({
                        'antecedent': f"{col1}={row[col1]}",
                        'consequent': f"{col2}={row[col2]}",
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })
    
    return association_rules
```

## Phase 3: Domain-Specific Analysis (Payments Focus)

### 3.1 Financial Transaction Patterns

```python
def analyze_payment_flows(transaction_table):
    """Analyze payment flow patterns"""
    df = spark.sql(f"SELECT * FROM {transaction_table}")
    
    # Identify potential amount columns
    amount_columns = [col_name for col_name, col_type in df.dtypes 
                     if col_type in ['double', 'decimal', 'float'] and 
                     any(keyword in col_name.lower() for keyword in ['amount', 'value', 'sum', 'total'])]
    
    # Identify potential currency columns
    currency_columns = [col_name for col_name in df.columns 
                       if any(keyword in col_name.lower() for keyword in ['currency', 'ccy', 'curr'])]
    
    # Identify potential account/entity columns
    entity_columns = [col_name for col_name in df.columns 
                     if any(keyword in col_name.lower() for keyword in ['account', 'customer', 'entity', 'party'])]
    
    payment_analysis = {}
    
    # Amount distribution analysis
    for amount_col in amount_columns:
        amount_stats = df.select(
            min(amount_col).alias('min'),
            max(amount_col).alias('max'),
            mean(amount_col).alias('avg'),
            expr(f"percentile_approx({amount_col}, 0.5)").alias('median'),
            expr(f"percentile_approx({amount_col}, 0.95)").alias('p95'),
            expr(f"percentile_approx({amount_col}, 0.99)").alias('p99')
        ).collect()[0]
        
        payment_analysis[f'{amount_col}_distribution'] = dict(amount_stats.asDict())
        
        # Large transaction analysis (potential wholesale vs retail)
        large_threshold = amount_stats['p95']
        large_transactions = df.filter(col(amount_col) > large_threshold).count()
        total_transactions = df.count()
        
        payment_analysis[f'{amount_col}_large_transactions'] = {
            'threshold': large_threshold,
            'count': large_transactions,
            'percentage': (large_transactions / total_transactions) * 100
        }
    
    # Currency pair analysis
    if currency_columns:
        for curr_col in currency_columns:
            currency_dist = df.groupBy(curr_col).count().orderBy(desc('count'))
            payment_analysis[f'{curr_col}_distribution'] = currency_dist.collect()
    
    # Transaction flow analysis
    if len(entity_columns) >= 2:
        # Assume first two are sender/receiver
        sender_col, receiver_col = entity_columns[0], entity_columns[1]
        
        # Network analysis
        flow_analysis = df.groupBy(sender_col, receiver_col).agg(
            count('*').alias('transaction_count'),
            sum(amount_columns[0]).alias('total_amount') if amount_columns else lit(0)
        ).orderBy(desc('transaction_count'))
        
        payment_analysis['transaction_flows'] = flow_analysis.limit(100).collect()
        
        # Identify hub entities (high degree centrality)
        sender_activity = df.groupBy(sender_col).count().orderBy(desc('count'))
        receiver_activity = df.groupBy(receiver_col).count().orderBy(desc('count'))
        
        payment_analysis['top_senders'] = sender_activity.limit(20).collect()
        payment_analysis['top_receivers'] = receiver_activity.limit(20).collect()
    
    return payment_analysis

def detect_settlement_patterns(transaction_table, status_column, date_column):
    """Detect settlement and reconciliation patterns"""
    df = spark.sql(f"SELECT * FROM {transaction_table}")
    
    # Status distribution
    status_dist = df.groupBy(status_column).count().orderBy(desc('count'))
    
    # Settlement timing analysis
    settlement_timing = df.filter(col(status_column).isin(['SETTLED', 'COMPLETED', 'SUCCESS'])) \
                         .groupBy(dayofweek(col(date_column)).alias('day_of_week')) \
                         .count() \
                         .orderBy('day_of_week')
    
    # Failed transaction analysis
    failed_transactions = df.filter(col(status_column).isin(['FAILED', 'REJECTED', 'ERROR']))
    failed_analysis = {
        'total_failed': failed_transactions.count(),
        'failure_rate': (failed_transactions.count() / df.count()) * 100
    }
    
    # Time to settlement analysis (if timestamps available)
    time_columns = [col_name for col_name in df.columns 
                   if any(keyword in col_name.lower() for keyword in ['time', 'timestamp', 'date'])]
    
    settlement_patterns = {
        'status_distribution': status_dist.collect(),
        'settlement_timing': settlement_timing.collect(),
        'failure_analysis': failed_analysis
    }
    
    return settlement_patterns
```

### 3.2 Regulatory & Compliance Pattern Detection

```python
def detect_compliance_patterns(tables_info):
    """Detect regulatory and compliance-related patterns"""
    compliance_analysis = {}
    
    for table_info in tables_info:
        table_name = table_info['table_name']
        columns = table_info['columns']
        
        # Look for compliance-related columns
        compliance_indicators = {
            'aml_screening': [col for col in columns if any(keyword in col.lower() 
                            for keyword in ['aml', 'screening', 'sanction', 'watch', 'pep'])],
            'reporting': [col for col in columns if any(keyword in col.lower() 
                        for keyword in ['report', 'regulatory', 'filing', 'declaration'])],
            'risk_scoring': [col for col in columns if any(keyword in col.lower() 
                           for keyword in ['risk', 'score', 'rating', 'level'])],
            'audit_trail': [col for col in columns if any(keyword in col.lower() 
                          for keyword in ['audit', 'trail', 'log', 'history', 'modified'])]
        }
        
        if any(compliance_indicators.values()):
            df = spark.sql(f"SELECT * FROM {table_name}")
            
            table_compliance = {}
            
            # AML/Sanctions screening analysis
            if compliance_indicators['aml_screening']:
                for aml_col in compliance_indicators['aml_screening']:
                    aml_dist = df.groupBy(aml_col).count().orderBy(desc('count'))
                    table_compliance[f'{aml_col}_distribution'] = aml_dist.collect()
            
            # Risk scoring analysis
            if compliance_indicators['risk_scoring']:
                for risk_col in compliance_indicators['risk_scoring']:
                    if dict(df.dtypes)[risk_col] in ['int', 'double', 'float']:
                        risk_stats = df.select(
                            min(risk_col).alias('min_risk'),
                            max(risk_col).alias('max_risk'),
                            mean(risk_col).alias('avg_risk')
                        ).collect()[0]
                        table_compliance[f'{risk_col}_statistics'] = dict(risk_stats.asDict())
            
            # Data retention analysis
            if compliance_indicators['audit_trail']:
                date_cols = [col for col in compliance_indicators['audit_trail'] 
                           if 'date' in col.lower() or 'time' in col.lower()]
                
                for date_col in date_cols:
                    retention_analysis = df.select(
                        min(col(date_col)).alias('earliest'),
                        max(col(date_col)).alias('latest'),
                        count('*').alias('total_records')
                    ).collect()[0]
                    
                    table_compliance[f'{date_col}_retention'] = dict(retention_analysis.asDict())
            
            compliance_analysis[table_name] = table_compliance
    
    return compliance_analysis

def identify_reporting_structures(tables_info):
    """Identify potential regulatory reporting structures"""
    reporting_tables = []
    
    for table_info in tables_info:
        table_name = table_info['table_name']
        columns = table_info['columns']
        
        # Look for reporting indicators
        reporting_keywords = ['report', 'filing', 'declaration', 'statement', 'summary']
        date_keywords = ['date', 'period', 'month', 'quarter', 'year']
        
        has_reporting_cols = any(keyword in table_name.lower() for keyword in reporting_keywords)
        has_date_cols = any(keyword in col.lower() for col in columns for keyword in date_keywords)
        
        if has_reporting_cols or (has_date_cols and 'summary' in table_name.lower()):
            df = spark.sql(f"SELECT * FROM {table_name}")
            
            # Analyze reporting frequency
            date_columns = [col for col in columns if any(keyword in col.lower() for keyword in date_keywords)]
            
            reporting_frequency = {}
            for date_col in date_columns:
                try:
                    freq_analysis = df.groupBy(
                        year(col(date_col)).alias('year'),
                        month(col(date_col)).alias('month')
                    ).count().orderBy('year', 'month')
                    
                    reporting_frequency[date_col] = freq_analysis.collect()
                except:
                    continue  # Skip if date parsing fails
            
            reporting_tables.append({
                'table_name': table_name,
                'reporting_frequency': reporting_frequency,
                'potential_type': 'regulatory_report' if has_reporting_cols else 'summary_report'
            })
    
    return reporting_tables
```

## Phase 4: Advanced Analytics for Business Logic Inference

### 4.1 Machine Learning for Business Rules

```python
def infer_approval_logic(transaction_table, status_column, feature_columns):
    """Use decision trees to infer approval/rejection logic"""
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    df = spark.sql(f"SELECT * FROM {transaction_table}")
    
    # Sample data for ML analysis
    sample_df = df.sample(0.1) if df.count() > 100000 else df
    pandas_df = sample_df.toPandas()
    
    # Prepare features
    feature_data = pandas_df[feature_columns].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in feature_data.columns:
        if feature_data[col].dtype == 'object':
            le = LabelEncoder()
            feature_data[col] = le.fit_transform(feature_data[col].astype(str))
            label_encoders[col] = le
    
    # Prepare target variable
    target = pandas_df[status_column]
    target_encoded = LabelEncoder().fit_transform(target)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, target_encoded, test_size=0.3, random_state=42
    )
    
    # Train decision tree
    dt_classifier = DecisionTreeClassifier(
        max_depth=10, 
        min_samples_split=100,
        random_state=42
    )
    dt_classifier.fit(X_train, y_train)
    
    # Extract rules
    tree_rules = export_text(dt_classifier, feature_names=feature_columns)
    
    # Feature importance
    feature_importance = dict(zip(feature_columns, dt_classifier.feature_importances_))
    
    return {
        'decision_rules': tree_rules,
        'feature_importance': feature_importance,
        'model_accuracy': dt_classifier.score(X_test, y_test),
        'label_encoders': label_encoders
    }

def sequence_analysis(transaction_table, entity_column, status_column, timestamp_column):
    """Analyze transaction lifecycle sequences"""
    df = spark.sql(f"SELECT * FROM {transaction_table}")
    
    # Get sequences for each transaction/entity
    sequences = df.select(entity_column, status_column, timestamp_column) \
                 .orderBy(entity_column, timestamp_column) \
                 .groupBy(entity_column) \
                 .agg(collect_list(status_column).alias('status_sequence'))
    
    # Analyze common sequences
    sequence_patterns = sequences.groupBy('status_sequence') \
                               .count() \
                               .orderBy(desc('count'))
    
    # Convert to pandas for more complex analysis
    patterns_pd = sequence_patterns.toPandas()
    
    # Find most common transaction flows
    common_flows = []
    for _, row in patterns_pd.head(20).iterrows():
        flow = ' -> '.join(row['status_sequence'])
        common_flows.append({
            'flow': flow,
            'count': row['count'],
            'sequence_length': len(row['status_sequence'])
        })
    
    return {
        'common_flows': common_flows,
        'total_unique_sequences': len(patterns_pd),
        'sequence_distribution': patterns_pd['count'].describe().to_dict()
    }
```

### 4.2 Process Mining Implementation

```python
def process_mining_analysis(event_log_table, case_id, activity, timestamp):
    """Apply process mining to discover business processes"""
    df = spark.sql(f"SELECT * FROM {event_log_table}")
    
    # Prepare event log
    event_log = df.select(
        col(case_id).alias('case_id'),
        col(activity).alias('activity'),
        col(timestamp).alias('timestamp')
    ).orderBy('case_id', 'timestamp')
    
    # Convert to pandas for process mining
    event_log_pd = event_log.toPandas()
    event_log_pd['timestamp'] = pd.to_datetime(event_log_pd['timestamp'])
    
    # Calculate process metrics
    process_metrics = {}
    
    # 1. Activity frequency
    activity_freq = event_log_pd['activity'].value_counts().to_dict()
    process_metrics['activity_frequency'] = activity_freq
    
    # 2. Case duration analysis
    case_durations = event_log_pd.groupby('case_id').agg({
        'timestamp': ['min', 'max']
    })
    case_durations.columns = ['start_time', 'end_time']
    case_durations['duration_hours'] = (
        case_durations['end_time'] - case_durations['start_time']
    ).dt.total_seconds() / 3600
    
    process_metrics['duration_stats'] = case_durations['duration_hours'].describe().to_dict()
    
    # 3. Transition analysis (directly-follows relationships)
    transitions = []
    for case in event_log_pd['case_id'].unique():
        case_events = event_log_pd[event_log_pd['case_id'] == case].sort_values('timestamp')
        activities = case_events['activity'].tolist()
        
        for i in range(len(activities) - 1):
            transitions.append({
                'from_activity': activities[i],
                'to_activity': activities[i + 1],
                'case_id': case
            })
    
    transition_df = pd.DataFrame(transitions)
    transition_freq = transition_df.groupby(['from_activity', 'to_activity']).size().reset_index(name='frequency')
    
    process_metrics['transitions'] = transition_freq.to_dict('records')
    
    # 4. Identify process variants
    variants = event_log_pd.groupby('case_id')['activity'].apply(
        lambda x: ' -> '.join(x)
    ).value_counts()
    
    process_metrics['process_variants'] = {
        'total_variants': len(variants),
        'top_variants': variants.head(10).to_dict()
    }
    
    # 5. Bottleneck analysis
    activity_times = event_log_pd.groupby(['case_id', 'activity'])['timestamp'].first().reset_index()
    activity_times = activity_times.sort_values(['case_id', 'timestamp'])
    
    # Calculate time between activities
    time_between_activities = []
    for case in activity_times['case_id'].unique():
        case_data = activity_times[activity_times['case_id'] == case]
        for i in range(len(case_data) - 1):
            time_diff = (case_data.iloc[i+1]['timestamp'] - case_data.iloc[i]['timestamp']).total_seconds() / 3600
            time_between_activities.append({
                'from_activity': case_data.iloc[i]['activity'],
                'to_activity': case_data.iloc[i+1]['activity'],
                'time_hours': time_diff
            })
    
    if time_between_activities:
        bottleneck_df = pd.DataFrame(time_between_activities)
        bottleneck_analysis = bottleneck_df.groupby(['from_activity', 'to_activity'])['time_hours'].agg([
            'mean', 'median', 'std', 'count'
        ]).reset_index()
        
        process_metrics['bottlenecks'] = bottleneck_analysis.nlargest(10, 'mean').to_dict('records')
    
    return process_metrics

def anomaly_detection_in_processes(event_log_table, case_id, activity, timestamp):
    """Detect anomalies in business processes"""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    
    df = spark.sql(f"SELECT * FROM {event_log_table}")
    event_log_pd = df.toPandas()
    
    # Feature engineering for anomaly detection
    case_features = []
    
    for case in event_log_pd[case_id].unique():
        case_data = event_log_pd[event_log_pd[case_id] == case].sort_values(timestamp)
        
        # Calculate case-level features
        features = {
            'case_id': case,
            'total_activities': len(case_data),
            'unique_activities': case_data[activity].nunique(),
            'case_duration_hours': (
                pd.to_datetime(case_data[timestamp].max()) - 
                pd.to_datetime(case_data[timestamp].min())
            ).total_seconds() / 3600,
            'activity_sequence_length': len(case_data[activity].tolist()),
            'repeated_activities': len(case_data) - case_data[activity].nunique()
        }
        
        case_features.append(features)
    
    features_df = pd.DataFrame(case_features)
    
    # Prepare features for anomaly detection
    feature_columns = ['total_activities', 'unique_activities', 'case_duration_hours', 
                      'activity_sequence_length', 'repeated_activities']
    
    X = features_df[feature_columns].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    
    # Identify anomalous cases
    features_df['is_anomaly'] = anomaly_labels == -1
    anomalous_cases = features_df[features_df['is_anomaly']]
    
    return {
        'total_cases': len(features_df),
        'anomalous_cases_count': len(anomalous_cases),
        'anomaly_percentage': (len(anomalous_cases) / len(features_df)) * 100,
        'anomalous_cases': anomalous_cases.to_dict('records'),
        'feature_statistics': features_df[feature_columns].describe().to_dict()
    }
```

## Phase 5: Knowledge Extraction & Documentation

### 5.1 Automated Documentation Generation

```python
def generate_data_dictionary(tables_info, profiling_results):
    """Generate comprehensive data dictionary"""
    data_dictionary = {}
    
    for table_info in tables_info:
        table_name = table_info['table_name']
        columns = table_info['columns']
        
        table_dict = {
            'table_description': infer_table_purpose(table_name, columns),
            'row_count': table_info['row_count'],
            'columns': {}
        }
        
        # Get profiling results for this table
        if table_name in profiling_results:
            profile = profiling_results[table_name]
            
            for col in columns:
                if col in profile.get('column_profiles', {}):
                    col_profile = profile['column_profiles'][col]
                    
                    column_description = {
                        'data_type': col_profile['data_type'],
                        'null_percentage': col_profile['null_percentage'],
                        'distinct_count': col_profile['distinct_count'],
                        'inferred_purpose': infer_column_purpose(col, col_profile),
                        'data_quality_issues': identify_data_quality_issues(col_profile),
                        'business_meaning': infer_business_meaning(col, col_profile)
                    }
                    
                    # Add type-specific information
                    if col_profile['data_type'] in ['int', 'bigint', 'double', 'float']:
                        column_description.update({
                            'range': f"{col_profile.get('min_value', 'N/A')} - {col_profile.get('max_value', 'N/A')}",
                            'outlier_percentage': col_profile.get('outlier_percentage', 0)
                        })
                    elif col_profile['data_type'] == 'string':
                        column_description.update({
                            'avg_length': col_profile.get('avg_length', 'N/A'),
                            'patterns': col_profile.get('patterns', {})
                        })
                    
                    table_dict['columns'][col] = column_description
        
        data_dictionary[table_name] = table_dict
    
    return data_dictionary

def infer_table_purpose(table_name, columns):
    """Infer the business purpose of a table"""
    name_lower = table_name.lower()
    
    # Payment-specific patterns
    if any(keyword in name_lower for keyword in ['transaction', 'payment', 'transfer']):
        return "Transaction/Payment processing table"
    elif any(keyword in name_lower for keyword in ['customer', 'client', 'party']):
        return "Customer/Entity master data"
    elif any(keyword in name_lower for keyword in ['account', 'balance']):
        return "Account information and balances"
    elif any(keyword in name_lower for keyword in ['rate', 'exchange', 'currency']):
        return "Exchange rate or pricing data"
    elif any(keyword in name_lower for keyword in ['audit', 'log', 'history']):
        return "Audit trail or historical data"
    elif any(keyword in name_lower for keyword in ['report', 'summary']):
        return "Reporting or summary data"
    elif any(keyword in name_lower for keyword in ['config', 'parameter', 'setting']):
        return "Configuration or parameter table"
    else:
        # Analyze columns for additional clues
        col_keywords = ' '.join(columns).lower()
        if 'amount' in col_keywords and 'date' in col_keywords:
            return "Financial transaction or movement table"
        elif 'status' in col_keywords and 'process' in col_keywords:
            return "Process or workflow status table"
        else:
            return "Purpose unknown - requires further analysis"

def infer_column_purpose(column_name, column_profile):
    """Infer the purpose of a column based on name and profile"""
    name_lower = column_name.lower()
    
    # Common ID patterns
    if any(keyword in name_lower for keyword in ['id', 'key', 'number']) and column_profile['cardinality_ratio'] > 0.8:
        return "Unique identifier or primary key"
    
    # Amount/value patterns
    elif any(keyword in name_lower for keyword in ['amount', 'value', 'total', 'sum']):
        return "Monetary amount or financial value"
    
    # Date/time patterns
    elif any(keyword in name_lower for keyword in ['date', 'time', 'timestamp']):
        return "Date or timestamp field"
    
    # Status patterns
    elif 'status' in name_lower and column_profile['distinct_count'] < 20:
        return "Status or state indicator"
    
    # Code patterns
    elif column_profile.get('patterns', {}).get('currency_code', {}).get('percentage', 0) > 80:
        return "Currency code (ISO 4217)"
    elif column_profile.get('patterns', {}).get('country_code', {}).get('percentage', 0) > 80:
        return "Country code (ISO 3166)"
    elif column_profile.get('patterns', {}).get('swift_code', {}).get('percentage', 0) > 50:
        return "SWIFT/BIC code"
    
    # Address patterns
    elif any(keyword in name_lower for keyword in ['address', 'street', 'city', 'zip', 'postal']):
        return "Address component"
    
    # Name patterns
    elif any(keyword in name_lower for keyword in ['name', 'description', 'desc']):
        return "Name or description field"
    
    else:
        return "Purpose requires domain expert review"

def identify_data_quality_issues(column_profile):
    """Identify potential data quality issues"""
    issues = []
    
    if column_profile['null_percentage'] > 50:
        issues.append(f"High null rate: {column_profile['null_percentage']:.1f}%")
    
    if column_profile.get('outlier_percentage', 0) > 10:
        issues.append(f"High outlier rate: {column_profile['outlier_percentage']:.1f}%")
    
    if column_profile['cardinality_ratio'] == 1 and column_profile['null_percentage'] == 0:
        issues.append("All values are unique (potential over-normalization)")
    
    if column_profile['distinct_count'] == 1:
        issues.append("Single value across all records (potential constant)")
    
    return issues if issues else ["No major data quality issues detected"]

def infer_business_meaning(column_name, column_profile):
    """Infer business meaning from patterns"""
    name_lower = column_name.lower()
    
    # Payment-specific business meanings
    business_meanings = {
        'sender': "Originating party of the payment",
        'receiver': "Beneficiary party of the payment",
        'amount': "Transaction monetary amount",
        'currency': "Currency denomination",
        'reference': "Transaction reference number",
        'status': "Processing status of transaction",
        'fee': "Processing fee or charge",
        'rate': "Exchange rate applied",
        'swift': "International bank identifier",
        'iban': "International bank account number",
        'routing': "Domestic bank routing number"
    }
    
    for keyword, meaning in business_meanings.items():
        if keyword in name_lower:
            return meaning
    
    return "Business meaning requires domain expert interpretation"
```

### 5.2 Process Flow Visualization

```python
def generate_process_flow_diagram(process_mining_results):
    """Generate process flow diagrams from process mining results"""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Create directed graph from transitions
    G = nx.DiGraph()
    
    transitions = process_mining_results.get('transitions', [])
    
    # Add nodes and edges
    for transition in transitions:
        from_activity = transition['from_activity']
        to_activity = transition['to_activity']
        frequency = transition['frequency']
        
        G.add_edge(from_activity, to_activity, weight=frequency)
    
    # Calculate node positions
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Draw nodes
    node_sizes = [G.in_degree(node) * 200 + 300 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7)
    
    # Draw edges with thickness based on frequency
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [w / max_weight * 5 for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          alpha=0.6, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Add edge labels for high-frequency transitions
    edge_labels = {(u, v): str(G[u][v]['weight']) 
                  for u, v in edges if G[u][v]['weight'] > max_weight * 0.1}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title("Payment Process Flow Diagram", size=16, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def create_data_lineage_map(tables_info, relationships):
    """Create data lineage visualization"""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Create graph for data lineage
    G = nx.DiGraph()
    
    # Add tables as nodes
    for table_info in tables_info:
        table_name = table_info['table_name']
        row_count = table_info['row_count']
        G.add_node(table_name, size=row_count)
    
    # Add relationships as edges
    for relationship in relationships:
        table1 = relationship['table1']
        table2 = relationship['table2']
        common_cols = len(relationship['common_columns'])
        
        if common_cols > 0:
            G.add_edge(table1, table2, weight=common_cols)
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    plt.figure(figsize=(12, 8))
    
    # Node sizes based on row count
    node_sizes = [G.nodes[node].get('size', 1000) / 1000 for node in G.nodes()]
    node_sizes = [max(size, 100) for size in node_sizes]  # Minimum size
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightgreen', alpha=0.7)
    
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, 
                          arrowsize=20, edge_color='blue')
    
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("Data Lineage Map", size=16, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return plt
```

### 5.3 Complete Execution Pipeline

```python
def execute_complete_analysis(database_name):
    """Execute the complete data archaeology pipeline"""
    print("Starting Payment Application Data Archaeology...")
    
    # Phase 1: Discovery
    print("\n=== Phase 1: Data Discovery ===")
    tables = discover_tables()
    tables_info = [analyze_table_structure(table.tableName) for table in tables]
    relationships = identify_relationships(tables_info)
    
    print(f"Discovered {len(tables_info)} tables")
    print(f"Identified {len(relationships)} potential relationships")
    
    # Phase 2: Profiling
    print("\n=== Phase 2: Data Profiling ===")
    profiling_results = {}
    for table_info in tables_info[:5]:  # Limit for demo
        table_name = table_info['table_name']
        print(f"Profiling {table_name}...")
        profiling_results[table_name] = comprehensive_data_profile(table_name)
    
    # Phase 3: Domain Analysis
    print("\n=== Phase 3: Domain-Specific Analysis ===")
    
    # Identify transaction tables
    transaction_tables = [t for t in tables_info if any(keyword in t['table_name'].lower() 
                         for keyword in ['transaction', 'payment', 'transfer'])]
    
    payment_analysis = {}
    for table_info in transaction_tables:
        table_name = table_info['table_name']
        print(f"Analyzing payment patterns in {table_name}...")
        payment_analysis[table_name] = analyze_payment_flows(table_name)
    
    # Phase 4: Advanced Analytics
    print("\n=== Phase 4: Advanced Analytics ===")
    ml_insights = {}
    
    if transaction_tables:
        # Process mining on the first transaction table
        main_txn_table = transaction_tables[0]['table_name']
        
        # Try to identify key columns
        columns = transaction_tables[0]['columns']
        case_id_col = next((col for col in columns if 'id' in col.lower()), columns[0])
        status_col = next((col for col in columns if 'status' in col.lower()), None)
        timestamp_col = next((col for col in columns if any(keyword in col.lower() 
                            for keyword in ['date', 'time', 'timestamp'])), None)
        
        if status_col and timestamp_col:
            print(f"Running process mining on {main_txn_table}...")
            ml_insights['process_mining'] = process_mining_analysis(
                main_txn_table, case_id_col, status_col, timestamp_col
            )
    
    # Phase 5: Documentation
    print("\n=== Phase 5: Knowledge Extraction ===")
    data_dictionary = generate_data_dictionary(tables_info, profiling_results)
    
    # Compile final report
    final_report = {
        'summary': {
            'total_tables': len(tables_info),
            'total_relationships': len(relationships),
            'transaction_tables': len(transaction_tables),
            'analysis_date': pd.Timestamp.now().isoformat()
        },
        'tables_info': tables_info,
        'relationships': relationships,
        'profiling_results': profiling_results,
        'payment_analysis': payment_analysis,
        'ml_insights': ml_insights,
        'data_dictionary': data_dictionary
    }
    
    print("\n=== Analysis Complete ===")
    print(f"Generated comprehensive analysis of {len(tables_info)} tables")
    print("Results include: data dictionary, relationship mapping, payment flow analysis, and process insights")
    
    return final_report

# Execute the analysis
if __name__ == "__main__":
    results = execute_complete_analysis("payments_db")
    
    # Save results
    import json
    with open('payment_app_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nAnalysis results saved to 'payment_app_analysis.json'")
```

This comprehensive implementation provides a complete data archaeology toolkit for analyzing your mainframe payment application data. Each phase builds upon the previous one, gradually reconstructing the business logic and operational patterns from the data itself.

The code is modular and can be adapted based on your specific data structures and requirements. Start with the discovery phase to understand your data landscape, then progressively apply more sophisticated analysis techniques as patterns emerge.
