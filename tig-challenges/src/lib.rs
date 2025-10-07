pub const BUILD_TIME_PATH: &str = env!("CARGO_MANIFEST_DIR");

macro_rules! conditional_pub {
    (fn $name:ident $($rest:tt)*) => {
        #[cfg(not(feature = "hide_verification"))]
        pub fn $name $($rest)*

        #[cfg(feature = "hide_verification")]
        fn $name $($rest)*
    };
}

macro_rules! impl_base64_serde {
    ($name:ident { $( $field:ident : $ty:ty ),* $(,)? }) => {
        paste::paste! {
            #[derive(Debug, Clone)]
            pub struct $name {
                $( pub $field : $ty ),*
            }

            #[derive(serde::Serialize, serde::Deserialize)]
            struct [<$name Data>] {
                $( $field : $ty ),*
            }

            impl serde::Serialize for $name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    use flate2::{write::GzEncoder, Compression};
                    use base64::engine::general_purpose::STANDARD as BASE64;
                    use base64::Engine;
                    use std::io::Write;

                    let helper = [<$name Data>] {
                        $( $field: self.$field.clone() ),*
                    };

                    let bincode_data = bincode::serialize(&helper)
                        .map_err(|e| serde::ser::Error::custom(format!("Bincode serialization failed: {}", e)))?;

                    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                    encoder
                        .write_all(&bincode_data)
                        .map_err(|e| serde::ser::Error::custom(format!("Compression failed: {}", e)))?;
                    let compressed_data = encoder
                        .finish()
                        .map_err(|e| serde::ser::Error::custom(format!("Compression finish failed: {}", e)))?;

                    let encoded = BASE64.encode(&compressed_data);
                    serializer.serialize_str(&encoded)
                }
            }

            impl<'de> serde::Deserialize<'de> for $name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de>,
                {
                    use flate2::read::GzDecoder;
                    use base64::engine::general_purpose::STANDARD as BASE64;
                    use base64::Engine;
                    use std::io::Read;
                    use std::fmt;

                    struct VisitorImpl;

                    impl<'de> serde::de::Visitor<'de> for VisitorImpl {
                        type Value = $name;

                        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            write!(f, "a base64 encoded, compressed, bincode serialized {}", stringify!($name))
                        }

                        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                        where
                            E: serde::de::Error,
                        {
                            let compressed = BASE64.decode(v)
                                .map_err(|e| E::custom(format!("Base64 decode failed: {}", e)))?;

                            let mut decoder = GzDecoder::new(&compressed[..]);
                            let mut decompressed = Vec::new();
                            decoder
                                .read_to_end(&mut decompressed)
                                .map_err(|e| E::custom(format!("Decompression failed: {}", e)))?;

                            let data: [<$name Data>] = bincode::deserialize(&decompressed)
                                .map_err(|e| E::custom(format!("Bincode deserialization failed: {}", e)))?;

                            Ok($name {
                                $( $field: data.$field ),*
                            })
                        }
                    }

                    deserializer.deserialize_str(VisitorImpl)
                }
            }
        }
    };
}

#[cfg(feature = "c001")]
pub mod satisfiability;
#[cfg(feature = "c001")]
pub use satisfiability as c001;
#[cfg(feature = "c002")]
pub mod vehicle_routing;
#[cfg(feature = "c002")]
pub use vehicle_routing as c002;
#[cfg(feature = "c003")]
pub mod knapsack;
#[cfg(feature = "c003")]
pub use knapsack as c003;
#[cfg(feature = "c004")]
pub mod vector_search;
#[cfg(feature = "c004")]
pub use vector_search as c004;
#[cfg(feature = "c005")]
pub mod hypergraph;
#[cfg(feature = "c005")]
pub use hypergraph as c005;
#[cfg(feature = "c006")]
pub(crate) mod neuralnet;
#[cfg(feature = "c006")]
pub mod neuralnet_optimizer;
#[cfg(feature = "c006")]
pub use neuralnet_optimizer as c006;
