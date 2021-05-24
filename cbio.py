from utils import *

def load_generic(fname=None, df=None):
	spark = get_spark_session()

	# Load from file
	if fname is not None:
		# Load data if in a single file
		if '.csv' in fname:
			df = spark.read.csv(fname, header=True)

		# Load data if in chunks
		else:
			df = spark.read.csv(fname + '/*.csv', header=True)
	
	# Load from pandas dataframe
	elif df is not None:
		df = spark.createDataFrame(df)
	
	# Issue with filename/dataframe provided
	else:
		print('No filename or pandas dataframe provided.')
		return ''
	
	return df

def check_cols(df, required_cols, error_msg):
	if not set(required_cols).intersection(set(df.columns)) == set(required_cols):
		raise ValueError(error_msg)

def check_colvalues(df, colname, colvalues, error_msg):
	if set(df.select(colname).distinct().rdd.map(lambda r: r[0]).collect()).union(set(colvalues)) != set(colvalues):
		raise ValueError(error_msg)
	

def load_cdr(fname=None, df=None, verify=True):
	spark = get_spark_session()

	cdr = load_generic(fname=fname, df=df)

	if verify:
	
		# Check that required columns are present
		required_cols = ['txn_type', 'caller_id', 'recipient_id', 'timestamp', 'duration']
		error_msg = 'CDR format incorrect. CDR must include the following columns: ' + ', '.join(required_cols)
		check_cols(cdr, required_cols, error_msg)

		# Check txn_type column
		error_msg = 'CDR format incorrect. Column txn_type can only include voice and sms.'
		check_colvalues(cdr, 'txn_type', ['voice', 'sms'], error_msg)
		
		# Clean international column
		error_msg = 'CDR format incorrect. Column international can only include domestic, international, and other.'
		check_colvalues(cdr, 'international', ['domestic', 'international', 'other'], error_msg)
	
	# Clean timestamp column
	cdr = cdr.withColumn('timestamp', to_timestamp(cdr['timestamp'], 'yyyy-MM-dd HH:mm:ss'))\
		.withColumn('day', date_trunc('day', col('timestamp')))
		
	# Clean duration column
	cdr = cdr.withColumn('duration', col('duration').cast('float'))
	
	return cdr

def load_topups(fname=None, df=None, verify=True):
	spark = get_spark_session()

	topups = load_generic(fname=fname, df=df)
		
	# Clean timestamp column
	topups = topups.withColumn('timestamp', to_timestamp(topups['timestamp'], 'yyyy-MM-dd HH:mm:ss'))\
		.withColumn('day', date_trunc('day', col('timestamp')))
		
	# Clean duration column
	topups = topups.withColumn('amount', col('amount').cast('float'))
		
	return topups

def load_mobiledata(fname=None, df=None, verify=True):
	spark = get_spark_session()

	mobiledata = load_generic(fname=fname, df=df)
		
	# Clean timestamp column
	mobiledata = mobiledata.withColumn('timestamp', to_timestamp(mobiledata['timestamp'], 'yyyy-MM-dd HH:mm:ss'))\
		.withColumn('day', date_trunc('day', col('timestamp')))
	
	# Clean duration column
	mobiledata = mobiledata.withColumn('volume', col('volume').cast('float'))
	
	return mobiledata

def load_mobilemoney(fname=None, df=None, verify=True):
	spark = get_spark_session()

	mobilemoney = load_generic(fname=fname, df=df)

	if verify:
	
		# Check that required columns are present
		required_cols = ['txn_type', 'caller_id', 'recipient_id', 'timestamp', 'amount']
		error_msg = 'Mobile money format incorrect. Mobile money records must include the following columns: ' + \
			', '.join(required_cols)
		check_cols(mobilemoney, required_cols, error_msg)
		
		# Check txn_type column
		txn_types = ['cashin', 'cashout', 'p2p', 'billpay', 'other']
		error_msg = 'Mobile money format incorrect. Column txn_type can only include ' + ', '.join(txn_types)
		check_colvalues(mobilemoney, 'txn_type', txn_types, error_msg)

	# Clean timestamp column
	mobilemoney = mobilemoney.withColumn('timestamp', to_timestamp(mobilemoney['timestamp'], 'yyyy-MM-dd HH:mm:ss'))\
		.withColumn('day', date_trunc('day', col('timestamp')))
	
	# Clean duration column
	mobilemoney = mobilemoney.withColumn('amount', col('amount').cast('float'))

	# Clean balance columns
	for c in mobilemoney.columns:
		if 'balance' in c:
			mobilemoney = mobilemoney.withColumn(c, col(c).cast('float'))
	
	return mobilemoney