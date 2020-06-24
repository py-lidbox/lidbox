class GlobalMeanStddevPooling1D extends tf.layers.Layer {
  static get className() {
    return 'GlobalMeanStddevPooling1D';
  }
  constructor(config) {
    super(config || {name: "stats_pooling"});
  }
  computeOutputShape(inputShape) {
    return [inputShape[0], 2 * inputShape[2]];
  }
  call(inputs) {
    const input = inputs[0];
    const timeAxis = 1;
    const mean = tf.mean(input, timeAxis);
    const stddev = tf.sqrt(tf.mean(tf.square(tf.sub(input, mean)), timeAxis));
    return tf.concat([mean, stddev], timeAxis);
  }
};
tf.serialization.registerClass(GlobalMeanStddevPooling1D);
